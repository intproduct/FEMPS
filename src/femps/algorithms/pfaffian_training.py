"""Deterministic AD training for the E1/E2 fixed-number Pfaffian benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import time
from pathlib import Path

import torch

from femps.devices import resolve_device
from femps.exterior import (
    agp_tensor,
    antisymmetry_residual,
    bivector_decomposition_length,
    particle_tt_ranks,
    pair_matrix_from_channels,
    real_skew_pair_decomposition,
)
from femps.hamiltonians import (
    agp_energy,
    antisymmetric_two_particle_hamiltonian,
    exact_interacting_pair_energy,
    harmonic_pair_hamiltonian,
)


@dataclass(frozen=True, slots=True)
class PfaffianPairConfig:
    """Complete E1/E2 single-AGP training configuration."""

    basis_order: int = 10
    kappa: float = 0.35
    omega: float = 1.0
    steps: int = 1000
    learning_rate: float = 3e-2
    final_learning_rate: float = 1e-5
    seed: int = 0
    device: str = "cpu"
    record_points: int = 20
    checkpoint_every: int = 100

    def validate(self) -> None:
        if min(
            self.basis_order,
            self.steps,
            self.record_points,
            self.checkpoint_every,
        ) < 1:
            raise ValueError("D, steps, record_points, and checkpoint_every must be positive")
        if self.basis_order < 2:
            raise ValueError("two spinless fermions require basis_order >= 2")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError("require 0 < final_learning_rate <= learning_rate")
        exact_interacting_pair_energy(kappa=self.kappa, omega=self.omega)


@dataclass(frozen=True, slots=True)
class FactorizedPairConfig:
    """Low pair-rank E2 scan configuration."""

    basis_order: int = 12
    pair_channels: int = 1
    kappa: float = 0.35
    omega: float = 1.0
    steps: int = 1000
    learning_rate: float = 2e-2
    final_learning_rate: float = 1e-5
    seed: int = 0
    device: str = "cpu"
    record_points: int = 20

    def validate(self) -> None:
        if min(
            self.basis_order,
            self.pair_channels,
            self.steps,
            self.record_points,
        ) < 1:
            raise ValueError("D, pair_channels, steps, and record_points must be positive")
        if 2 * self.pair_channels > self.basis_order:
            raise ValueError("pair_channels cannot exceed floor(D/2)")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError("require 0 < final_learning_rate <= learning_rate")
        exact_interacting_pair_energy(kappa=self.kappa, omega=self.omega)


def _random_complex_matrix(order: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(order, order, generator=generator, dtype=torch.float64)
    imag = torch.randn(order, order, generator=generator, dtype=torch.float64)
    return torch.complex(real, imag).to(device)


def _normalized_raw(raw: torch.Tensor) -> torch.Tensor:
    skew = raw - raw.transpose(0, 1)
    norm = torch.linalg.vector_norm(skew)
    if norm == 0:
        raise ValueError("initial skew pair matrix is zero")
    return 0.5 * skew / norm


def _save_checkpoint(
    path: Path,
    *,
    config: PfaffianPairConfig,
    step: int,
    raw: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    history: list[dict[str, float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "config": asdict(config),
            "step": step,
            "raw": raw.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
        },
        path,
    )


def run_pfaffian_pair(
    config: PfaffianPairConfig,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    max_steps_this_call: int | None = None,
) -> dict:
    """Optimize E1/E2 and return a JSON-serializable record.

    Projection of the skew matrix to unit Frobenius norm happens after each
    optimizer step and outside the loss graph, matching latticeTN's accepted
    Rayleigh-quotient training convention.
    """

    config.validate()
    if max_steps_this_call is not None and max_steps_this_call < 1:
        raise ValueError("max_steps_this_call must be positive")
    device = resolve_device(config.device)
    one_body, two_body = harmonic_pair_hamiltonian(
        config.basis_order,
        kappa=config.kappa,
        omega=config.omega,
        dtype=torch.complex128,
        device=device,
    )
    resumed = False
    if resume:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise ValueError("resume requires an existing checkpoint_path")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload["config"] != asdict(config):
            raise ValueError("checkpoint configuration does not match requested run")
        raw = torch.nn.Parameter(payload["raw"].to(device))
        start_step = int(payload["step"])
        history = list(payload["history"])
        resumed = True
    else:
        torch.manual_seed(config.seed)
        raw = torch.nn.Parameter(
            _normalized_raw(
                _random_complex_matrix(config.basis_order, config.seed, device)
            )
        )
        start_step = 0
        history = []

    optimizer = torch.optim.Adam([raw], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.final_learning_rate,
    )
    if resumed:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])

    def pair_matrix() -> torch.Tensor:
        return raw - raw.transpose(0, 1)

    def energy() -> torch.Tensor:
        return agp_energy(pair_matrix(), 1, one_body, two_body)

    record_interval = max(1, config.steps // config.record_points)
    if not history:
        history.append(
            {
                "step": 0,
                "energy": float(energy().detach().cpu()),
                "learning_rate": config.learning_rate,
            }
        )
    stop_step = config.steps
    if max_steps_this_call is not None:
        stop_step = min(config.steps, start_step + max_steps_this_call)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(start_step + 1, stop_step + 1):
        optimizer.zero_grad()
        loss = energy()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            raw.copy_(_normalized_raw(raw))
        scheduler.step()
        if step % record_interval == 0 or step == stop_step:
            history.append(
                {
                    "step": step,
                    "energy": float(energy().detach().cpu()),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )
        if checkpoint_path is not None and (
            step % config.checkpoint_every == 0 or step == stop_step
        ):
            _save_checkpoint(
                checkpoint_path,
                config=config,
                step=step,
                raw=raw,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final_pair = pair_matrix().detach()
    final_energy = float(energy().detach().cpu())
    continuum_reference = exact_interacting_pair_energy(
        kappa=config.kappa, omega=config.omega
    )
    truncated_hamiltonian = antisymmetric_two_particle_hamiltonian(
        one_body.detach(), two_body
    )
    truncated_reference = float(
        torch.linalg.eigvalsh(truncated_hamiltonian).real[0].detach().cpu()
    )
    completed = stop_step == config.steps
    if completed and final_energy < truncated_reference - 1e-8:
        raise RuntimeError("variational energy fell below the truncated exact reference")
    explicit_state = agp_tensor(final_pair.cpu(), 1)
    return {
        "schema_version": 1,
        "experiment": "functional_pfaffian_harmonic_pair",
        "benchmark": "E1" if config.kappa == 0 else "E2",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "compute_capability": (
                torch.cuda.get_device_capability(device) if device.type == "cuda" else None
            ),
        },
        "resumed": resumed,
        "completed": completed,
        "completed_steps": stop_step,
        "continuum_reference_energy": continuum_reference,
        "truncated_reference_energy": truncated_reference,
        "initial_energy": history[0]["energy"],
        "final_energy": final_energy,
        "error_vs_continuum": abs(final_energy - continuum_reference),
        "error_vs_truncated": abs(final_energy - truncated_reference),
        "variational_margin_truncated": final_energy - truncated_reference,
        "pair_decomposition_length": bivector_decomposition_length(explicit_state),
        "ordinary_particle_tt_ranks": particle_tt_ranks(explicit_state),
        "antisymmetry_residual": float(antisymmetry_residual(explicit_state)),
        "energy_history": history,
        "elapsed_seconds_this_call": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }


def _project_factor_channels(
    left: torch.Tensor,
    right: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Gauge-balance channels without changing their pair matrix, then scale it."""

    with torch.no_grad():
        left_norms = torch.linalg.vector_norm(left, dim=1).clamp_min(1e-14)
        right_norms = torch.linalg.vector_norm(right, dim=1).clamp_min(1e-14)
        left.div_(left_norms[:, None])
        right.div_(right_norms[:, None])
        weights.mul_(left_norms * right_norms)
        pair = pair_matrix_from_channels(left, right, weights)
        pair_norm = torch.linalg.vector_norm(pair)
        if pair_norm <= 1e-14:
            raise RuntimeError("factorized pair matrix collapsed to zero")
        weights.div_(pair_norm)


def run_factorized_pfaffian_pair(
    config: FactorizedPairConfig,
    *,
    initial_pair_matrix: torch.Tensor | None = None,
) -> dict:
    """Optimize an E1/E2 state constrained to at most ``pair_channels`` wedges."""

    config.validate()
    device = resolve_device(config.device)
    one_body, two_body = harmonic_pair_hamiltonian(
        config.basis_order,
        kappa=config.kappa,
        omega=config.omega,
        dtype=torch.complex128,
        device=device,
    )
    if initial_pair_matrix is None:
        left_initial = _random_complex_matrix(
            config.basis_order, config.seed, device
        )[: config.pair_channels]
        right_initial = _random_complex_matrix(
            config.basis_order, config.seed + 1009, device
        )[: config.pair_channels]
        weight_initial = (
            torch.ones(
                config.pair_channels, dtype=torch.complex128, device=device
            )
            / config.pair_channels
        )
        initialization = "random_channels"
    else:
        if initial_pair_matrix.shape != (config.basis_order, config.basis_order):
            raise ValueError("initial_pair_matrix shape does not match basis_order")
        left_initial, right_initial, weight_initial = real_skew_pair_decomposition(
            initial_pair_matrix.to(device), config.pair_channels
        )
        initialization = "truncated_real_skew_reference"
    left = torch.nn.Parameter(left_initial)
    right = torch.nn.Parameter(right_initial)
    weights = torch.nn.Parameter(weight_initial)
    _project_factor_channels(left, right, weights)
    parameters = [left, right, weights]
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.final_learning_rate,
    )

    def pair_matrix() -> torch.Tensor:
        return pair_matrix_from_channels(left, right, weights)

    def energy() -> torch.Tensor:
        return agp_energy(pair_matrix(), 1, one_body, two_body)

    record_interval = max(1, config.steps // config.record_points)
    history = [
        {
            "step": 0,
            "energy": float(energy().detach().cpu()),
            "learning_rate": config.learning_rate,
        }
    ]
    best_energy = float(history[0]["energy"])
    best_parameters = tuple(
        parameter.detach().clone() for parameter in parameters
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        loss = energy()
        loss.backward()
        optimizer.step()
        _project_factor_channels(left, right, weights)
        scheduler.step()
        if step % record_interval == 0 or step == config.steps:
            recorded_energy = float(energy().detach().cpu())
            history.append(
                {
                    "step": step,
                    "energy": recorded_energy,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )
            if recorded_energy < best_energy:
                best_energy = recorded_energy
                best_parameters = tuple(
                    parameter.detach().clone() for parameter in parameters
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    terminal_energy = float(energy().detach().cpu())
    with torch.no_grad():
        for parameter, best in zip(parameters, best_parameters):
            parameter.copy_(best)
    final_pair = pair_matrix().detach()
    final_energy = float(energy().detach().cpu())
    truncated_reference = float(
        torch.linalg.eigvalsh(
            antisymmetric_two_particle_hamiltonian(one_body.detach(), two_body)
        )[0]
        .real.detach()
        .cpu()
    )
    continuum_reference = exact_interacting_pair_energy(
        kappa=config.kappa, omega=config.omega
    )
    state = agp_tensor(final_pair.cpu(), 1)
    return {
        "schema_version": 1,
        "experiment": "functional_factorized_pfaffian_harmonic_pair",
        "config": asdict(config),
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "initial_energy": history[0]["energy"],
        "initialization": initialization,
        "final_energy": final_energy,
        "terminal_energy_before_best_restore": terminal_energy,
        "best_recorded_energy": best_energy,
        "continuum_reference_energy": continuum_reference,
        "truncated_reference_energy": truncated_reference,
        "error_vs_continuum": abs(final_energy - continuum_reference),
        "error_vs_truncated": abs(final_energy - truncated_reference),
        "variational_margin_truncated": final_energy - truncated_reference,
        "requested_pair_channels": config.pair_channels,
        "observed_pair_decomposition_length": bivector_decomposition_length(state),
        "ordinary_particle_tt_ranks": particle_tt_ranks(state),
        "antisymmetry_residual": float(antisymmetry_residual(state)),
        "energy_history": history,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
