"""Variable-projection training for finite sums of fixed-number AGPs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
from pathlib import Path
import time

import torch

from femps.devices import resolve_device
from femps.exterior import (
    agp_exterior_coefficients,
    agp_tensor,
    antisymmetry_residual,
    particle_tt_ranks,
)
from femps.hamiltonians import (
    agp_hamiltonian_transition_matrices,
    antisymmetric_many_body_hamiltonian,
    exact_interacting_harmonic_fermion_energy,
    harmonic_pair_hamiltonian,
    soft_coulomb_operator,
)

from .agp_subspace import GeneralizedEigenResult, solve_generalized_hermitian


@dataclass(frozen=True, slots=True)
class FiniteAgpConfig:
    """Configuration for finite-AGP variable-projection training."""

    basis_order: int = 8
    particles: int = 4
    agp_terms: int = 2
    kappa: float = 0.35
    omega: float = 1.0
    steps: int = 600
    learning_rate: float = 5e-3
    final_learning_rate: float = 1e-5
    seed: int = 0
    device: str = "cpu"
    record_points: int = 20
    checkpoint_every: int = 100
    overlap_relative_threshold: float = 1e-10
    frozen_prefix_terms: int = 0
    interaction_model: str = "harmonic"
    soft_coulomb_coupling: float = 1.0
    soft_coulomb_softening: float = 1.0
    soft_coulomb_quadrature_order: int = 128
    soft_coulomb_relative_threshold: float = 1e-14

    def validate(self) -> None:
        if self.particles < 2 or self.particles % 2:
            raise ValueError("finite AGP training requires a positive even particle count")
        if self.basis_order < self.particles:
            raise ValueError("basis_order must be at least the particle count")
        if min(
            self.agp_terms,
            self.steps,
            self.record_points,
            self.checkpoint_every,
        ) < 1:
            raise ValueError("K, steps, record_points, and checkpoint_every must be positive")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError("require 0 < final_learning_rate <= learning_rate")
        if self.overlap_relative_threshold < 0:
            raise ValueError("overlap_relative_threshold must be nonnegative")
        if not 0 <= self.frozen_prefix_terms < self.agp_terms:
            raise ValueError("frozen_prefix_terms must satisfy 0 <= frozen < K")
        if self.interaction_model == "harmonic":
            exact_interacting_harmonic_fermion_energy(
                self.particles, kappa=self.kappa, omega=self.omega
            )
        elif self.interaction_model == "soft_coulomb":
            if self.soft_coulomb_quadrature_order < 1:
                raise ValueError("soft-Coulomb quadrature order must be positive")
            if self.soft_coulomb_coupling < 0 or self.soft_coulomb_softening <= 0:
                raise ValueError("invalid soft-Coulomb coupling or softening")
        else:
            raise ValueError("interaction_model must be harmonic or soft_coulomb")


def _build_hamiltonian(
    config: FiniteAgpConfig, device: torch.device | str
):
    one_body = harmonic_pair_hamiltonian(
        config.basis_order,
        kappa=0.0,
        omega=config.omega,
        dtype=torch.complex128,
        device=device,
    )[0]
    if config.interaction_model == "harmonic":
        _, interaction = harmonic_pair_hamiltonian(
            config.basis_order,
            kappa=config.kappa,
            omega=config.omega,
            dtype=torch.complex128,
            device=device,
        )
        return one_body, None if config.kappa == 0 else interaction
    interaction, _ = soft_coulomb_operator(
        config.basis_order,
        quadrature_order=config.soft_coulomb_quadrature_order,
        coupling=config.soft_coulomb_coupling,
        softening=config.soft_coulomb_softening,
        relative_threshold=config.soft_coulomb_relative_threshold,
        dtype=torch.complex128,
        device=device,
    )
    return one_body, interaction


def _random_complex_raw(
    terms: int, dimension: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(
        terms, dimension, dimension, generator=generator, dtype=torch.float64
    )
    imaginary = torch.randn(
        terms, dimension, dimension, generator=generator, dtype=torch.float64
    )
    return torch.complex(real, imaginary).to(device)


def _project_raw_pair_scales(raw: torch.Tensor) -> None:
    """Keep raw skew parts at unit norm; the physical forward map is unchanged."""

    with torch.no_grad():
        skew = raw - raw.transpose(1, 2)
        norms = torch.linalg.vector_norm(skew, dim=(1, 2))
        if torch.any(norms <= 1e-14):
            raise RuntimeError("an AGP pair matrix collapsed to zero")
        raw.div_(norms[:, None, None])


def canonical_pair_matrices(raw: torch.Tensor) -> torch.Tensor:
    """Return unit-norm skew matrices with a deterministic continuous phase gauge."""

    if raw.ndim != 3 or raw.shape[1] != raw.shape[2]:
        raise ValueError("raw pair parameters must have shape (K,D,D)")
    skew = raw - raw.transpose(1, 2)
    norms = torch.linalg.vector_norm(skew, dim=(1, 2))
    if torch.any(norms <= torch.finfo(norms.dtype).tiny):
        raise ValueError("every raw pair matrix must have nonzero skew part")
    normalized = skew / norms[:, None, None]
    upper = torch.triu_indices(
        raw.shape[1], raw.shape[2], offset=1, device=raw.device
    )
    upper_values = normalized[:, upper[0], upper[1]]
    anchors = torch.argmax(torch.abs(upper_values).detach(), dim=1)
    anchor_values = upper_values[
        torch.arange(raw.shape[0], device=raw.device), anchors
    ]
    phases = anchor_values / torch.abs(anchor_values)
    return normalized * phases.conj()[:, None, None]


def _canonical_output_order(pair_matrices: torch.Tensor) -> torch.Tensor:
    """Return a deterministic discrete AGP order for artifacts and restarts."""

    upper = torch.triu_indices(
        pair_matrices.shape[1], pair_matrices.shape[2], offset=1
    )
    values = pair_matrices.detach().cpu()[:, upper[0], upper[1]]
    keys = []
    for term in range(pair_matrices.shape[0]):
        magnitudes = torch.abs(values[term])
        anchor = int(torch.argmax(magnitudes))
        keys.append((anchor, -float(magnitudes[anchor]), term))
    order = [item[2] for item in sorted(keys)]
    return torch.tensor(order, dtype=torch.long, device=pair_matrices.device)


def _solve_state(
    raw: torch.Tensor,
    *,
    pairs: int,
    one_body: torch.Tensor,
    interaction,
    overlap_relative_threshold: float,
) -> tuple[torch.Tensor, GeneralizedEigenResult]:
    pair_matrices = canonical_pair_matrices(raw)
    overlap, hamiltonian = agp_hamiltonian_transition_matrices(
        pair_matrices, pairs, one_body, interaction
    )
    result = solve_generalized_hermitian(
        hamiltonian,
        overlap,
        relative_threshold=overlap_relative_threshold,
    )
    return pair_matrices, result


def _history_entry(step: int, result: GeneralizedEigenResult, learning_rate: float) -> dict:
    return {
        "step": step,
        "energy": float(result.energy.detach().cpu()),
        "learning_rate": learning_rate,
        "overlap_eigenvalues": [
            float(value) for value in result.overlap_eigenvalues.detach().cpu()
        ],
        "raw_overlap_eigenvalues": [
            float(value)
            for value in result.raw_overlap_eigenvalues.detach().cpu()
        ],
        "retained_rank": result.retained_rank,
        "discarded_rank": result.discarded_rank,
        "retained_condition_number": result.retained_condition_number,
        "raw_overlap_condition_number": result.raw_overlap_condition_number,
        "generalized_residual_norm": float(result.residual_norm.detach().cpu()),
    }


def _save_checkpoint(
    path: Path,
    *,
    config: FiniteAgpConfig,
    step: int,
    raw: torch.Tensor,
    best_raw: torch.Tensor,
    best_energy: float,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    history: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "config": asdict(config),
            "step": step,
            "raw": raw.detach().cpu(),
            "best_raw": best_raw.detach().cpu(),
            "best_energy": best_energy,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
        },
        path,
    )


def run_finite_agp_variable_projection(
    config: FiniteAgpConfig,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    max_steps_this_call: int | None = None,
    initial_pair_matrices: torch.Tensor | None = None,
) -> dict:
    """Optimize nonlinear pair matrices while solving amplitudes exactly."""

    config.validate()
    if max_steps_this_call is not None and max_steps_this_call < 1:
        raise ValueError("max_steps_this_call must be positive")
    if resume and initial_pair_matrices is not None:
        raise ValueError("resume cannot be combined with initial_pair_matrices")
    device = resolve_device(config.device)
    pairs = config.particles // 2
    one_body, interaction = _build_hamiltonian(config, device)
    resumed = False
    if resume:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise ValueError("resume requires an existing checkpoint_path")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload["config"] != asdict(config):
            raise ValueError("checkpoint configuration does not match requested run")
        raw = torch.nn.Parameter(payload["raw"].to(device))
        best_raw = payload["best_raw"].to(device)
        best_energy = float(payload["best_energy"])
        start_step = int(payload["step"])
        history = list(payload["history"])
        initialization = "resumed_random"
        resumed = True
    else:
        if initial_pair_matrices is None:
            initial_raw = _random_complex_raw(
                config.agp_terms,
                config.basis_order,
                config.seed,
                device,
            )
            initialization = "blind_random"
        else:
            if initial_pair_matrices.shape != (
                config.agp_terms,
                config.basis_order,
                config.basis_order,
            ):
                raise ValueError("initial_pair_matrices shape does not match config")
            initial_raw = 0.5 * initial_pair_matrices.to(
                dtype=torch.complex128, device=device
            )
            initialization = "provided_pair_matrices"
        raw = torch.nn.Parameter(initial_raw)
        _project_raw_pair_scales(raw)
        start_step = 0
        history = []
        _, initial_result = _solve_state(
            raw,
            pairs=pairs,
            one_body=one_body,
            interaction=interaction,
            overlap_relative_threshold=config.overlap_relative_threshold,
        )
        best_energy = float(initial_result.energy.detach().cpu())
        best_raw = raw.detach().clone()

    optimizer = torch.optim.Adam([raw], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.final_learning_rate,
    )
    if resumed:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])

    if not history:
        _, initial_result = _solve_state(
            raw,
            pairs=pairs,
            one_body=one_body,
            interaction=interaction,
            overlap_relative_threshold=config.overlap_relative_threshold,
        )
        history.append(
            _history_entry(0, initial_result, config.learning_rate)
        )
    stop_step = config.steps
    if max_steps_this_call is not None:
        stop_step = min(config.steps, start_step + max_steps_this_call)
    record_interval = max(1, config.steps // config.record_points)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(start_step + 1, stop_step + 1):
        optimizer.zero_grad()
        _, result = _solve_state(
            raw,
            pairs=pairs,
            one_body=one_body,
            interaction=interaction,
            overlap_relative_threshold=config.overlap_relative_threshold,
        )
        result.energy.backward()
        if config.frozen_prefix_terms:
            assert raw.grad is not None
            raw.grad[: config.frozen_prefix_terms].zero_()
        optimizer.step()
        _project_raw_pair_scales(raw)
        scheduler.step()
        if step % record_interval == 0 or step == stop_step:
            _, recorded = _solve_state(
                raw,
                pairs=pairs,
                one_body=one_body,
                interaction=interaction,
                overlap_relative_threshold=config.overlap_relative_threshold,
            )
            entry = _history_entry(
                step, recorded, scheduler.get_last_lr()[0]
            )
            history.append(entry)
            if entry["energy"] < best_energy:
                best_energy = entry["energy"]
                best_raw = raw.detach().clone()
        if checkpoint_path is not None and (
            step % config.checkpoint_every == 0 or step == stop_step
        ):
            _save_checkpoint(
                checkpoint_path,
                config=config,
                step=step,
                raw=raw,
                best_raw=best_raw,
                best_energy=best_energy,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    _, terminal_result = _solve_state(
        raw,
        pairs=pairs,
        one_body=one_body,
        interaction=interaction,
        overlap_relative_threshold=config.overlap_relative_threshold,
    )
    with torch.no_grad():
        raw.copy_(best_raw)
    final_pairs, final_result = _solve_state(
        raw,
        pairs=pairs,
        one_body=one_body,
        interaction=interaction,
        overlap_relative_threshold=config.overlap_relative_threshold,
    )
    order = _canonical_output_order(final_pairs)
    final_pairs = final_pairs[order]
    final_amplitudes = final_result.amplitudes[order]

    one_body_cpu, interaction_cpu = _build_hamiltonian(config, "cpu")
    truth_hamiltonian = antisymmetric_many_body_hamiltonian(
        one_body_cpu, config.particles, interaction_cpu
    )
    truth_values, truth_vectors = torch.linalg.eigh(truth_hamiltonian)
    finite_reference = float(truth_values[0].real)
    coefficients = sum(
        final_amplitudes[term].detach().cpu()
        * agp_exterior_coefficients(final_pairs[term].detach().cpu(), pairs)
        for term in range(config.agp_terms)
    )
    coefficient_norm = torch.vdot(coefficients, coefficients).real
    explicit_energy = float(
        (
            torch.vdot(coefficients, truth_hamiltonian @ coefficients)
            / coefficient_norm
        ).real
    )
    fidelity = float(
        (
            torch.abs(torch.vdot(truth_vectors[:, 0], coefficients)) ** 2
            / coefficient_norm
        ).real
    )
    continuum_reference = (
        exact_interacting_harmonic_fermion_energy(
            config.particles, kappa=config.kappa, omega=config.omega
        )
        if config.interaction_model == "harmonic"
        else None
    )
    final_energy = float(final_result.energy.detach().cpu())
    completed = stop_step == config.steps
    if completed and final_energy < finite_reference - 1e-8:
        raise RuntimeError("variational energy fell below finite-basis truth")
    explicit_diagnostics = config.basis_order**config.particles <= 2_000_000
    particle_state = None
    if explicit_diagnostics:
        particle_state = sum(
            final_amplitudes[term].detach().cpu()
            * agp_tensor(final_pairs[term].detach().cpu(), pairs)
            for term in range(config.agp_terms)
        )
    return {
        "schema_version": 1,
        "experiment": "finite_agp_harmonic_variable_projection",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "compute_capability": (
                torch.cuda.get_device_capability(device)
                if device.type == "cuda"
                else None
            ),
        },
        "initialization": initialization,
        "resumed": resumed,
        "completed": completed,
        "completed_steps": stop_step,
        "initial_energy": history[0]["energy"],
        "terminal_energy_before_best_restore": float(
            terminal_result.energy.detach().cpu()
        ),
        "final_energy": final_energy,
        "explicit_exterior_energy": explicit_energy,
        "polynomial_explicit_absolute_difference": abs(
            final_energy - explicit_energy
        ),
        "finite_basis_reference_energy": finite_reference,
        "continuum_reference_energy": continuum_reference,
        "basis_error_vs_continuum": (
            finite_reference - continuum_reference
            if continuum_reference is not None
            else None
        ),
        "error_vs_finite_basis": final_energy - finite_reference,
        "error_vs_continuum": (
            final_energy - continuum_reference
            if continuum_reference is not None
            else None
        ),
        "finite_basis_ground_fidelity": fidelity,
        "final_overlap_eigenvalues": [
            float(value)
            for value in final_result.overlap_eigenvalues.detach().cpu()
        ],
        "final_raw_overlap_eigenvalues": [
            float(value)
            for value in final_result.raw_overlap_eigenvalues.detach().cpu()
        ],
        "final_retained_rank": final_result.retained_rank,
        "final_discarded_rank": final_result.discarded_rank,
        "final_retained_condition_number": final_result.retained_condition_number,
        "final_raw_overlap_condition_number": (
            final_result.raw_overlap_condition_number
        ),
        "final_generalized_residual_norm": float(
            final_result.residual_norm.detach().cpu()
        ),
        "ordinary_particle_tt_ranks": (
            list(particle_tt_ranks(particle_state))
            if particle_state is not None
            else None
        ),
        "antisymmetry_residual": (
            float(antisymmetry_residual(particle_state))
            if particle_state is not None
            else None
        ),
        "history": history,
        "elapsed_seconds_this_call": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
