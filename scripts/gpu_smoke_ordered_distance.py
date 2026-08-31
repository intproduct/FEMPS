"""CPU/Blackwell parity for native ordered-distance latticeTN contraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from femps.baselines.ordered_distance_mpo import (
    gap_soft_coulomb_hamiltonian_mpo,
    ordered_values_to_gap_mps,
)
from femps.devices import resolve_device
from femps.ordered_distance import gap_hamiltonian, gap_values_to_ordered_values


def _evaluate(base_cores, device: torch.device, grid_points: int, particles: int, spacing: float):
    from latticetn.mps import MPS

    cores = [core.to(device) for core in base_cores]
    mps = MPS.from_tensors(
        cores,
        dtype=cores[0].dtype,
        device=device,
        requires_grad=True,
    )
    mpo = gap_soft_coulomb_hamiltonian_mpo(
        grid_points,
        particles,
        spacing,
        dtype=cores[0].dtype,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    energy = mps.energy_with_MPO(mpo)
    energy.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    gradients = [core.grad.detach().cpu() for core in mps.tensors]
    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    return energy.detach().cpu(), gradients, seconds, peak


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-points", type=int, default=8)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=0.7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase15_ordered_distance_gpu_parity.json"
        ),
    )
    arguments = parser.parse_args()
    gpu = resolve_device("auto")
    if gpu.type != "cuda":
        raise RuntimeError("no CUDA device is available")
    hamiltonian = gap_hamiltonian(
        arguments.grid_points,
        arguments.particles,
        arguments.spacing,
        soft_coulomb=True,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
    ordered_values = gap_values_to_ordered_values(
        eigenvectors[:, 0], arguments.grid_points, arguments.particles
    )
    base_mps, ranks, discarded = ordered_values_to_gap_mps(
        ordered_values,
        arguments.grid_points,
        arguments.particles,
        requires_grad=False,
    )
    base_cores = [core.detach().clone() for core in base_mps.tensors]
    cpu_energy, cpu_gradients, cpu_seconds, _ = _evaluate(
        base_cores,
        torch.device("cpu"),
        arguments.grid_points,
        arguments.particles,
        arguments.spacing,
    )
    gpu_energy, gpu_gradients, gpu_seconds, gpu_peak = _evaluate(
        base_cores,
        gpu,
        arguments.grid_points,
        arguments.particles,
        arguments.spacing,
    )
    result = {
        "schema_version": 1,
        "experiment": "phase15_ordered_distance_cpu_gpu_parity",
        "grid_points": arguments.grid_points,
        "particles": arguments.particles,
        "spacing": arguments.spacing,
        "dtype": "float64",
        "device": str(gpu),
        "gpu": torch.cuda.get_device_name(gpu),
        "compute_capability": list(torch.cuda.get_device_capability(gpu)),
        "exact_mps_ranks": list(ranks),
        "tt_svd_discarded_norm": float(discarded),
        "truth_energy": float(eigenvalues[0]),
        "cpu_energy": float(cpu_energy),
        "gpu_energy": float(gpu_energy),
        "energy_absolute_difference": float(torch.abs(cpu_energy - gpu_energy)),
        "gradient_max_absolute_difference": max(
            float(torch.max(torch.abs(cpu - accelerated)))
            for cpu, accelerated in zip(cpu_gradients, gpu_gradients, strict=True)
        ),
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
        "gpu_peak_memory_bytes": gpu_peak,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if (
        result["energy_absolute_difference"] < 1e-11
        and result["gradient_max_absolute_difference"] < 1e-10
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
