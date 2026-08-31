"""Deterministic AD training harness for the 2201 functional-MPS baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import platform
import time

import torch

from femps.devices import resolve_device

from .coupled_oscillators import exact_ground_energy, functional_mps_energy
from .functional_mps import random_functional_mps


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Complete numerical configuration for one reproducible optimization."""

    num_oscillators: int = 4
    basis_order: int = 8
    bond_dimension: int = 16
    gamma: float = -0.5
    omega: float = 1.0
    steps: int = 1500
    learning_rate: float = 1e-2
    final_learning_rate: float = 1e-5
    seed: int = 0
    device: str = "cpu"
    record_points: int = 20

    def validate(self) -> None:
        if min(
            self.num_oscillators,
            self.basis_order,
            self.bond_dimension,
            self.steps,
            self.record_points,
        ) < 1:
            raise ValueError("N, D, chi, steps, and record_points must be positive")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError("require 0 < final_learning_rate <= learning_rate")
        exact_ground_energy(
            self.num_oscillators, gamma=self.gamma, omega=self.omega
        )


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def run_baseline(config: BaselineConfig) -> dict:
    """Optimize one functional MPS and return a JSON-serializable record."""
    config.validate()
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    mps = random_functional_mps(
        config.num_oscillators,
        config.basis_order,
        config.bond_dimension,
        dtype=torch.complex128,
        device=device,
        seed=config.seed,
    )
    optimizer = torch.optim.Adam(list(mps.parameters()), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.final_learning_rate,
    )
    reference = exact_ground_energy(
        config.num_oscillators, gamma=config.gamma, omega=config.omega
    )
    record_interval = max(1, config.steps // config.record_points)

    def measured_energy() -> float:
        value = functional_mps_energy(
            mps, gamma=config.gamma, omega=config.omega
        )
        return float(value.detach().cpu())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    initial = measured_energy()
    history = [
        {"step": 0, "energy": initial, "learning_rate": config.learning_rate}
    ]

    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        energy = functional_mps_energy(
            mps, gamma=config.gamma, omega=config.omega
        )
        energy.backward()
        optimizer.step()
        # Each core rescaling changes only the global state amplitude. This is
        # the latticeTN post-step stabilization and is outside the loss path.
        with torch.no_grad():
            for tensor in mps.tensors:
                norm = tensor.norm()
                if norm > 0:
                    tensor.div_(norm)
        scheduler.step()
        if step % record_interval == 0 or step == config.steps:
            history.append(
                {
                    "step": step,
                    "energy": measured_energy(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

    final = measured_energy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    best = min(item["energy"] for item in history)
    variational_margin = final - reference
    if variational_margin < -1e-8:
        raise RuntimeError(
            "variational energy fell below the exact continuum energy by "
            f"{-variational_margin:.3e}"
        )

    parameter_count = sum(parameter.numel() for parameter in mps.parameters())
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "schema_version": 1,
        "experiment": "2201_two_body_functional_mps",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "latticetn": _package_version("latticetn"),
            "femps": _package_version("femps"),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "compute_capability": (
                torch.cuda.get_device_capability(device) if device.type == "cuda" else None
            ),
        },
        "parameter_count": parameter_count,
        "peak_cuda_memory_bytes": peak_memory,
        "reference_energy": reference,
        "initial_energy": initial,
        "final_energy": final,
        "absolute_error": abs(final - reference),
        "variational_margin": variational_margin,
        "best_recorded_energy": best,
        "best_recorded_absolute_error": abs(best - reference),
        "energy_history": history,
        "elapsed_seconds": elapsed,
    }

