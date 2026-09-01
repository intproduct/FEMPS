"""Variable-projection optimizer for diagonal-path FEMPS.

Nonlinear orbital matrices are kept on a per-determinant Stiefel gauge through
QR.  Linear determinant amplitudes are solved exactly in their conditioned
overlap span at every step.  Full exterior materialization is used only for a
bounded independent truth audit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import platform
from pathlib import Path
import time

import torch

from femps.benchmarks import ProcessRSSMonitor
from femps.devices import resolve_device
from femps.exterior import (
    antisymmetry_residual,
    diagonal_path_exterior_coefficients,
    diagonal_path_hamiltonian_matrices,
    diagonal_path_structural_counts,
    diagonal_path_transition_diagnostics,
    exterior_coefficients_to_tensor,
    particle_tt_ranks,
)
from femps.hamiltonians import (
    FactorizedTwoBodyOperator,
    antisymmetric_many_body_hamiltonian,
    exact_interacting_harmonic_fermion_energy,
    harmonic_pair_hamiltonian,
    soft_coulomb_operator,
)

from .agp_subspace import GeneralizedEigenResult, solve_generalized_hermitian
from .diagonal_path_contract import (
    DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION,
    DIAGONAL_PATH_RESULT_SCHEMA_VERSION,
    load_diagonal_path_checkpoint,
    validate_diagonal_path_result,
)


@dataclass(frozen=True, slots=True)
class DiagonalPathConfig:
    """Configuration for a structured ``K``-Slater FEMPS optimization."""

    basis_order: int = 6
    particles: int = 2
    terms: int = 2
    kappa: float = 0.35
    omega: float = 1.0
    steps: int = 300
    learning_rate: float = 1e-2
    final_learning_rate: float = 1e-5
    seed: int = 0
    device: str = "cpu"
    record_points: int = 20
    checkpoint_every: int = 100
    overlap_relative_threshold: float = 1e-10
    truth_maximum_dimension: int = 300
    particle_tensor_maximum_coefficients: int = 100_000
    interaction_model: str = "harmonic"
    soft_coulomb_coupling: float = 1.0
    soft_coulomb_softening: float = 1.0
    soft_coulomb_quadrature_order: int = 64
    soft_coulomb_relative_threshold: float = 1e-13
    lbfgs_refinement_steps: int = 0
    lbfgs_learning_rate: float = 0.5

    def validate(self) -> None:
        if self.particles < 1 or self.basis_order < self.particles:
            raise ValueError("require D >= N >= 1")
        if min(
            self.terms,
            self.steps,
            self.record_points,
            self.checkpoint_every,
            self.truth_maximum_dimension,
            self.particle_tensor_maximum_coefficients,
        ) < 1:
            raise ValueError("K, step counts, and truth limits must be positive")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError("require 0 < final_learning_rate <= learning_rate")
        if self.overlap_relative_threshold < 0:
            raise ValueError("overlap_relative_threshold must be nonnegative")
        if self.lbfgs_refinement_steps < 0 or self.lbfgs_learning_rate <= 0:
            raise ValueError("invalid LBFGS refinement configuration")
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


def canonical_slater_orbitals(raw: torch.Tensor) -> torch.Tensor:
    """Return QR-gauged orthonormal orbitals for every determinant term."""

    if raw.ndim != 3 or raw.shape[1] < raw.shape[2]:
        raise ValueError("raw orbitals must have shape (K,D,N) with D >= N")
    if not (raw.is_floating_point() or raw.is_complex()):
        raise ValueError("raw orbitals must use a floating or complex dtype")
    q, r = torch.linalg.qr(raw, mode="reduced")
    diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
    magnitude = torch.abs(diagonal)
    if torch.any(magnitude <= torch.finfo(magnitude.dtype).tiny):
        raise ValueError("raw orbital matrix is rank deficient")
    phase = diagonal / magnitude
    return q * phase.conj()[:, None, :]


def embed_diagonal_path_orbitals(
    orbitals: torch.Tensor, target_basis_order: int
) -> torch.Tensor:
    """Embed a state exactly in a larger nested functional basis by zero padding."""

    if orbitals.ndim != 3:
        raise ValueError("orbitals must have shape (K,D,N)")
    if target_basis_order < orbitals.shape[1]:
        raise ValueError("target basis cannot be smaller than the source basis")
    embedded = orbitals.new_zeros(
        (orbitals.shape[0], target_basis_order, orbitals.shape[2])
    )
    embedded[:, : orbitals.shape[1], :] = orbitals
    return embedded


def extend_diagonal_path_terms(
    orbitals: torch.Tensor, target_terms: int, *, seed: int
) -> torch.Tensor:
    """Nest a ``K``-term state in a larger blind determinant span.

    Existing determinants are preserved exactly. Additional determinants use
    seeded random orthonormal orbitals, so variable projection can retain the
    source state without using a truth eigenvector to choose the new terms.
    """

    if orbitals.ndim != 3 or orbitals.shape[1] < orbitals.shape[2]:
        raise ValueError("orbitals must have shape (K,D,N) with D >= N")
    if target_terms < orbitals.shape[0]:
        raise ValueError("target term count cannot be smaller than the source")
    if target_terms == orbitals.shape[0]:
        return orbitals.clone()
    if not (orbitals.is_floating_point() or orbitals.is_complex()):
        raise ValueError("orbitals must use a floating or complex dtype")

    generator = torch.Generator().manual_seed(seed)
    extra_shape = (
        target_terms - orbitals.shape[0],
        orbitals.shape[1],
        orbitals.shape[2],
    )
    real = torch.randn(extra_shape, generator=generator, dtype=torch.float64)
    if orbitals.is_complex():
        imaginary = torch.randn(extra_shape, generator=generator, dtype=torch.float64)
        extra = torch.complex(real, imaginary)
    else:
        extra = real
    extra = canonical_slater_orbitals(
        extra.to(dtype=orbitals.dtype, device=orbitals.device)
    )
    return torch.cat((orbitals, extra), dim=0)


def _random_initial_orbitals(config: DiagonalPathConfig, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(config.seed)
    shape = (config.terms, config.basis_order, config.particles)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64)
    raw = torch.complex(real, imaginary)
    raw[0] = 0
    raw[0, : config.particles, :] = torch.eye(
        config.particles, dtype=torch.complex128
    )
    return raw.to(device)


def _project_raw_orbitals(raw: torch.Tensor) -> None:
    with torch.no_grad():
        raw.copy_(canonical_slater_orbitals(raw))


def _build_hamiltonian(
    config: DiagonalPathConfig, device: torch.device
) -> tuple[torch.Tensor, FactorizedTwoBodyOperator | None, dict]:
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
        return one_body, (None if config.kappa == 0 else interaction), {
            "model": "harmonic",
            "kappa": config.kappa,
            "omega": config.omega,
        }
    interaction, diagnostics = soft_coulomb_operator(
        config.basis_order,
        quadrature_order=config.soft_coulomb_quadrature_order,
        coupling=config.soft_coulomb_coupling,
        softening=config.soft_coulomb_softening,
        relative_threshold=config.soft_coulomb_relative_threshold,
        dtype=torch.complex128,
        device=device,
    )
    return one_body, interaction, {
        "model": "soft_coulomb",
        "coupling": config.soft_coulomb_coupling,
        "softening": config.soft_coulomb_softening,
        "quadrature_order": config.soft_coulomb_quadrature_order,
        "operator_factor_rank": diagnostics.retained_rank,
        "operator_factorization_error": diagnostics.dense_relative_factorization_error,
    }


def _provided_hamiltonian(
    config: DiagonalPathConfig,
    operators: tuple[torch.Tensor, FactorizedTwoBodyOperator | None],
    device: torch.device,
) -> tuple[torch.Tensor, FactorizedTwoBodyOperator | None, dict]:
    one_body, interaction = operators
    if one_body.shape != (config.basis_order, config.basis_order):
        raise ValueError("provided one-body operator has the wrong shape")
    if interaction is not None and interaction.dimension != config.basis_order:
        raise ValueError("provided two-body operator has the wrong dimension")
    moved_interaction = (
        FactorizedTwoBodyOperator(
            interaction.left.to(dtype=torch.complex128, device=device),
            interaction.right.to(dtype=torch.complex128, device=device),
            interaction.weights.to(dtype=torch.complex128, device=device),
        )
        if interaction is not None
        else None
    )
    return (
        one_body.to(dtype=torch.complex128, device=device),
        moved_interaction,
        {"model": "provided"},
    )


def _solve_state(
    raw: torch.Tensor,
    one_body: torch.Tensor,
    interaction: FactorizedTwoBodyOperator | None,
    relative_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, GeneralizedEigenResult]:
    orbitals = canonical_slater_orbitals(raw)
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        orbitals,
        one_body,
        two_body_left=(interaction.left if interaction is not None else None),
        two_body_right=(interaction.right if interaction is not None else None),
        two_body_weights=(interaction.weights if interaction is not None else None),
    )
    result = solve_generalized_hermitian(
        hamiltonian, overlap, relative_threshold=relative_threshold
    )
    return orbitals, overlap, hamiltonian, result


def _history_entry(
    step: int,
    result: GeneralizedEigenResult,
    learning_rate: float,
    *,
    optimizer_name: str = "adam",
) -> dict:
    return {
        "step": step,
        "optimizer": optimizer_name,
        "energy": float(result.energy.detach().cpu()),
        "learning_rate": learning_rate,
        "retained_rank": result.retained_rank,
        "discarded_rank": result.discarded_rank,
        "retained_condition_number": result.retained_condition_number,
        "raw_overlap_condition_number": result.raw_overlap_condition_number,
        "generalized_residual_norm": float(result.residual_norm.detach().cpu()),
    }


def _save_checkpoint(
    path: Path,
    *,
    config: DiagonalPathConfig,
    operator_id: str | None,
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
            "schema_version": DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION,
            "config": asdict(config),
            "operator_id": operator_id,
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


def _run_diagonal_path_variable_projection_impl(
    config: DiagonalPathConfig,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    max_steps_this_call: int | None = None,
    initial_orbitals: torch.Tensor | None = None,
    operators: tuple[torch.Tensor, FactorizedTwoBodyOperator | None] | None = None,
    operator_id: str | None = None,
) -> dict:
    """Optimize orbitals while solving determinant amplitudes exactly."""

    config.validate()
    if max_steps_this_call is not None and max_steps_this_call < 1:
        raise ValueError("max_steps_this_call must be positive")
    if resume and initial_orbitals is not None:
        raise ValueError("resume cannot be combined with initial_orbitals")
    if (operators is None and operator_id is not None) or (
        operators is not None and not operator_id
    ):
        raise ValueError("provided operators require a nonempty operator_id")
    device = resolve_device(config.device)
    one_body, interaction, operator_metadata = (
        _build_hamiltonian(config, device)
        if operators is None
        else _provided_hamiltonian(config, operators, device)
    )
    factor_rank = interaction.rank if interaction is not None else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if resume:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise ValueError("resume requires an existing checkpoint_path")
        payload = load_diagonal_path_checkpoint(
            checkpoint_path,
            map_location=device,
            expected_config=asdict(config),
            expected_operator_id=operator_id,
            verify_operator_id=True,
        )
        raw = torch.nn.Parameter(payload["raw"].to(device).contiguous())
        best_raw = payload["best_raw"].to(device)
        best_energy = float(payload["best_energy"])
        start_step = int(payload["step"])
        history = list(payload["history"])
        initialization = "resumed"
    else:
        expected_shape = (config.terms, config.basis_order, config.particles)
        if initial_orbitals is None:
            initial_raw = _random_initial_orbitals(config, device)
            initialization = "slater_plus_blind_random"
        else:
            if initial_orbitals.shape != expected_shape:
                raise ValueError("initial_orbitals shape does not match config")
            initial_raw = initial_orbitals.to(dtype=torch.complex128, device=device)
            initialization = "provided_orbitals"
        raw = torch.nn.Parameter(initial_raw.contiguous())
        _project_raw_orbitals(raw)
        start_step = 0
        history = []
        _, _, _, initial_result = _solve_state(
            raw,
            one_body,
            interaction,
            config.overlap_relative_threshold,
        )
        best_energy = float(initial_result.energy.detach().cpu())
        best_raw = raw.detach().clone()

    # Batched determinant/solve adjoints may return a non-contiguous view.
    # Adam accepts it, while torch.optim.LBFGS still flattens with ``view``.
    # Normalize the layout at the parameter boundary without changing values.
    raw.register_hook(lambda gradient: gradient.contiguous())

    optimizer = torch.optim.Adam([raw], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.steps, eta_min=config.final_learning_rate
    )
    if resume:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])

    stop_step = min(
        config.steps,
        start_step + (
            max_steps_this_call
            if max_steps_this_call is not None
            else config.steps - start_step
        ),
    )
    record_every = max(1, config.steps // config.record_points)
    started = time.perf_counter()
    if not history:
        _, _, _, initial_result = _solve_state(
            raw,
            one_body,
            interaction,
            config.overlap_relative_threshold,
        )
        history.append(
            _history_entry(0, initial_result, optimizer.param_groups[0]["lr"])
        )

    for step in range(start_step + 1, stop_step + 1):
        optimizer.zero_grad(set_to_none=True)
        _, _, _, result = _solve_state(
            raw,
            one_body,
            interaction,
            config.overlap_relative_threshold,
        )
        result.energy.backward()
        torch.nn.utils.clip_grad_norm_([raw], max_norm=10.0)
        optimizer.step()
        _project_raw_orbitals(raw)
        scheduler.step()

        with torch.no_grad():
            _, _, _, evaluated = _solve_state(
                raw,
                one_body,
                interaction,
                config.overlap_relative_threshold,
            )
            energy = float(evaluated.energy.detach().cpu())
            if energy < best_energy:
                best_energy = energy
                best_raw = raw.detach().clone()
            if step % record_every == 0 or step == stop_step:
                history.append(
                    _history_entry(step, evaluated, optimizer.param_groups[0]["lr"])
                )
        if checkpoint_path is not None and (
            step % config.checkpoint_every == 0 or step == stop_step
        ):
            _save_checkpoint(
                checkpoint_path,
                config=config,
                operator_id=operator_id,
                step=step,
                raw=raw,
                best_raw=best_raw,
                best_energy=best_energy,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
            )

    completed = stop_step == config.steps
    refinement = {
        "optimizer": "lbfgs",
        "requested_steps": config.lbfgs_refinement_steps,
        "closure_calls": 0,
        "initial_energy": best_energy,
        "final_energy": best_energy,
        "accepted": False,
    }
    if completed and config.lbfgs_refinement_steps:
        with torch.no_grad():
            raw.copy_(best_raw)
        lbfgs = torch.optim.LBFGS(
            [raw],
            lr=config.lbfgs_learning_rate,
            max_iter=config.lbfgs_refinement_steps,
            tolerance_grad=1e-10,
            tolerance_change=1e-13,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            solved = _solve_state(
                raw,
                one_body,
                interaction,
                config.overlap_relative_threshold,
            )[3]
            solved.energy.backward()
            refinement["closure_calls"] += 1
            return solved.energy

        lbfgs.step(closure)
        with torch.no_grad():
            evaluated = _solve_state(
                raw,
                one_body,
                interaction,
                config.overlap_relative_threshold,
            )[3]
            refined_energy = float(evaluated.energy.detach().cpu())
            refinement["final_energy"] = refined_energy
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_raw = raw.detach().clone()
                refinement["accepted"] = True
            history.append(
                _history_entry(
                    stop_step,
                    evaluated,
                    config.lbfgs_learning_rate,
                    optimizer_name="lbfgs",
                )
            )
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path,
                config=config,
                operator_id=operator_id,
                step=stop_step,
                raw=raw,
                best_raw=best_raw,
                best_energy=best_energy,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
            )
    elapsed = time.perf_counter() - started
    final_raw = best_raw if completed else raw.detach()
    with torch.no_grad():
        orbitals, overlap, hamiltonian, final_result = _solve_state(
            final_raw,
            one_body,
            interaction,
            config.overlap_relative_threshold,
        )
        amplitudes = final_result.amplitudes
        norm = torch.vdot(amplitudes, overlap @ amplitudes).real
        final_energy = float(final_result.energy.detach().cpu())

    exterior_dimension = math.comb(config.basis_order, config.particles)
    explicit_energy = None
    energy_variance = None
    finite_basis_reference = None
    materialized_residual = None
    ordinary_ranks = None
    if exterior_dimension <= config.truth_maximum_dimension:
        coefficients = diagonal_path_exterior_coefficients(orbitals, amplitudes)
        truth_hamiltonian = antisymmetric_many_body_hamiltonian(
            one_body, config.particles, interaction
        )
        finite_basis_reference = float(
            torch.linalg.eigvalsh(truth_hamiltonian)[0].detach().cpu()
        )
        acted = truth_hamiltonian @ coefficients
        coefficient_norm = torch.vdot(coefficients, coefficients).real
        explicit_energy_tensor = (
            torch.vdot(coefficients, acted) / coefficient_norm
        ).real
        explicit_energy = float(explicit_energy_tensor.detach().cpu())
        residual_vector = acted - explicit_energy_tensor * coefficients
        energy_variance = float(
            (torch.vdot(residual_vector, residual_vector).real / coefficient_norm)
            .detach()
            .cpu()
        )
        if config.basis_order**config.particles <= (
            config.particle_tensor_maximum_coefficients
        ):
            particle_state = exterior_coefficients_to_tensor(
                coefficients, config.basis_order, config.particles
            )
            materialized_residual = float(
                antisymmetry_residual(particle_state).detach().cpu()
            )
            ordinary_ranks = list(particle_tt_ranks(particle_state))

    continuum_reference = (
        exact_interacting_harmonic_fermion_energy(
            config.particles, kappa=config.kappa, omega=config.omega
        )
        if config.interaction_model == "harmonic"
        else None
    )
    return {
        "schema_version": DIAGONAL_PATH_RESULT_SCHEMA_VERSION,
        "method": "diagonal_path_femps",
        "evidence_level": "numerical",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "operator": operator_metadata,
        "operator_id": operator_id or "config_default",
        "initialization": initialization,
        "resumed": resume,
        "completed": completed,
        "completed_steps": stop_step,
        "energy": final_energy,
        "continuum_reference_energy": continuum_reference,
        "error_vs_continuum": (
            final_energy - continuum_reference
            if continuum_reference is not None
            else None
        ),
        "explicit_exterior_energy": explicit_energy,
        "finite_basis_reference_energy": finite_basis_reference,
        "error_vs_finite_basis": (
            final_energy - finite_basis_reference
            if finite_basis_reference is not None
            else None
        ),
        "polynomial_explicit_absolute_difference": (
            abs(final_energy - explicit_energy)
            if explicit_energy is not None
            else None
        ),
        "energy_variance": energy_variance,
        "energy_uncertainty": None,
        "norm": float(norm.detach().cpu()),
        "norm_error": float(abs(norm.detach().cpu() - 1.0)),
        "structural_antisymmetry_residual": 0.0,
        "materialized_antisymmetry_residual": materialized_residual,
        "ordinary_particle_tt_ranks": ordinary_ranks,
        "retained_rank": final_result.retained_rank,
        "discarded_rank": final_result.discarded_rank,
        "retained_condition_number": final_result.retained_condition_number,
        "raw_overlap_condition_number": final_result.raw_overlap_condition_number,
        "generalized_residual_norm": float(
            final_result.residual_norm.detach().cpu()
        ),
        "structural_counts": diagonal_path_structural_counts(
            config.particles, config.basis_order, config.terms, factor_rank
        ),
        "transition_diagnostics": diagonal_path_transition_diagnostics(orbitals),
        "history": history,
        "refinement": refinement,
        "elapsed_seconds_this_call": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }


def run_diagonal_path_variable_projection(
    config: DiagonalPathConfig,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    max_steps_this_call: int | None = None,
    initial_orbitals: torch.Tensor | None = None,
    operators: tuple[torch.Tensor, FactorizedTwoBodyOperator | None] | None = None,
    operator_id: str | None = None,
) -> dict:
    """Run the solver while measuring total wall time and sampled process RSS."""

    total_started = time.perf_counter()
    with ProcessRSSMonitor() as memory_monitor:
        result = _run_diagonal_path_variable_projection_impl(
            config,
            checkpoint_path=checkpoint_path,
            resume=resume,
            max_steps_this_call=max_steps_this_call,
            initial_orbitals=initial_orbitals,
            operators=operators,
            operator_id=operator_id,
        )
    memory_record = memory_monitor.record()
    result["optimization_elapsed_seconds_this_call"] = result[
        "elapsed_seconds_this_call"
    ]
    result["total_elapsed_seconds_this_call"] = time.perf_counter() - total_started
    result["cpu_memory"] = memory_record.as_dict()
    result["peak_cpu_rss_bytes"] = memory_record.peak_rss_bytes
    result["peak_cpu_rss_delta_bytes"] = memory_record.peak_delta_rss_bytes
    validate_diagonal_path_result(result)
    return result
