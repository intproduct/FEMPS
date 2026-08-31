"""CPU/Blackwell parity check for polynomial Pfaffian/AGP contractions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from femps.devices import resolve_device
from femps.exterior import agp_norm_generating, agp_one_body_expectation


def _random_complex(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imag = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imag)


def _evaluate(raw: torch.Tensor, operator: torch.Tensor, pairs: int) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    pair_matrix = (raw - raw.transpose(0, 1)) / raw.shape[0] ** 0.5
    norm = agp_norm_generating(pair_matrix, pairs)
    one_body = agp_one_body_expectation(pair_matrix, pairs, operator)
    energy = (one_body / norm).real
    energy.backward()
    if raw.device.type == "cuda":
        torch.cuda.synchronize(raw.device)
    return energy, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/gate_a_agp_gpu_parity.json"),
    )
    args = parser.parse_args()
    gpu = resolve_device("auto")
    if gpu.type != "cuda":
        raise RuntimeError("no CUDA device is available")
    raw_cpu = _random_complex((args.dimension, args.dimension), 401).requires_grad_(True)
    operator_raw = _random_complex((args.dimension, args.dimension), 402)
    operator_cpu = (operator_raw + operator_raw.conj().transpose(0, 1)) / args.dimension**0.5
    raw_gpu = raw_cpu.detach().to(gpu).requires_grad_(True)
    operator_gpu = operator_cpu.to(gpu)

    cpu_energy, cpu_seconds = _evaluate(raw_cpu, operator_cpu, args.pairs)
    gpu_energy, gpu_seconds = _evaluate(raw_gpu, operator_gpu, args.pairs)
    result = {
        "schema_version": 1,
        "experiment": "gate_a_agp_cpu_gpu_parity",
        "D": args.dimension,
        "N": 2 * args.pairs,
        "pairs": args.pairs,
        "dtype": "complex128",
        "device": str(gpu),
        "gpu": torch.cuda.get_device_name(gpu),
        "compute_capability": torch.cuda.get_device_capability(gpu),
        "cpu_energy": float(cpu_energy.detach()),
        "gpu_energy": float(gpu_energy.detach().cpu()),
        "energy_abs_diff": float((cpu_energy.detach() - gpu_energy.detach().cpu()).abs()),
        "gradient_max_abs_diff": float(
            (raw_cpu.grad - raw_gpu.grad.cpu()).abs().max()
        ),
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["energy_abs_diff"] < 1e-10 and result["gradient_max_abs_diff"] < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
