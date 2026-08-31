"""Benchmark the three exact FEMPS norm routes on controlled small systems."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch

from femps.exterior import (
    exterior_dynamic_program_cost,
    femps_norm_exterior,
    femps_norm_paths,
    materialize_femps_matrix,
)


def random_cores(
    particles: int, dimension: int, chi: int, seed: int
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    bonds = (1,) + (chi,) * (particles - 1) + (1,)
    cores = []
    for site in range(particles):
        shape = (bonds[site], dimension, bonds[site + 1])
        real = torch.randn(shape, generator=generator, dtype=torch.float64)
        imag = torch.randn(shape, generator=generator, dtype=torch.float64)
        cores.append(torch.complex(real, imag) / math.sqrt(dimension * chi))
    return cores


def full_norm(cores: list[torch.Tensor]) -> torch.Tensor:
    state = materialize_femps_matrix(cores)
    return torch.vdot(state.reshape(-1), state.reshape(-1)).real


def measure(function, cores: list[torch.Tensor], repeats: int) -> tuple[float, float]:
    elapsed = []
    value = None
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = function(cores)
        elapsed.append(time.perf_counter() - start)
    assert value is not None
    return statistics.median(elapsed), float(value.real.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=7)
    parser.add_argument("--chi", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/gate_a_norm_scaling.json"),
    )
    args = parser.parse_args()
    records = []
    for particles in range(args.n_min, args.n_max + 1):
        dimension = 2 * particles
        bonds = (1,) + (args.chi,) * (particles - 1) + (1,)
        cores = random_cores(particles, dimension, args.chi, 100 + particles)
        operations, exterior_peak = exterior_dynamic_program_cost(dimension, bonds)
        path_count = args.chi ** (particles - 1)
        methods = {
            "exterior": femps_norm_exterior,
            "paths": femps_norm_paths,
        }
        if particles <= 4:
            methods["full_tensor"] = full_norm
        timings = {}
        values = {}
        for name, function in methods.items():
            print(f"N={particles} D={dimension} chi={args.chi} method={name}", flush=True)
            timings[name], values[name] = measure(function, cores, args.repeats)
        reference = values["exterior"]
        relative_disagreement = {
            name: abs(value - reference) / max(abs(reference), 1.0)
            for name, value in values.items()
        }
        records.append(
            {
                "N": particles,
                "D": dimension,
                "chi": args.chi,
                "timing_seconds_median": timings,
                "norm_values": values,
                "relative_disagreement": relative_disagreement,
                "structural_counts": {
                    "full_tensor_coefficients": dimension**particles,
                    "virtual_paths": path_count,
                    "path_pair_determinants": path_count**2,
                    "exterior_dp_multiply_adds": operations,
                    "exterior_dp_peak_coefficients": exterior_peak,
                    "final_exterior_coefficients": math.comb(dimension, particles),
                },
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "gate_a_exact_norm_scaling",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "complex128",
        },
        "scan": vars(args) | {"output": str(args.output)},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
