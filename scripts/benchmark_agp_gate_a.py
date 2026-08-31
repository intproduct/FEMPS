"""Benchmark polynomial fixed-number Pfaffian/AGP contractions."""

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
    agp_norm_generating,
    agp_one_body_expectation,
    agp_structural_counts,
    agp_two_body_expectation_factorized,
)


def elementary_symmetric(values: torch.Tensor, degree: int) -> torch.Tensor:
    coefficients = torch.zeros(degree + 1, dtype=values.dtype, device=values.device)
    coefficients[0] = 1.0
    for value in values:
        updated = coefficients.clone()
        upper = min(degree, values.numel())
        for order in range(upper, 0, -1):
            updated[order] = coefficients[order] + value * coefficients[order - 1]
        coefficients = updated
    return coefficients[degree]


def dense_canonical_pair_matrix(dimension: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    channels = dimension // 2
    weights = torch.linspace(0.55, 0.85, channels, dtype=torch.float64)
    canonical = torch.zeros(dimension, dimension, dtype=torch.float64)
    for channel, weight in enumerate(weights):
        canonical[2 * channel, 2 * channel + 1] = weight
        canonical[2 * channel + 1, 2 * channel] = -weight
    generator = torch.Generator().manual_seed(seed)
    rotation, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    )
    dense = rotation @ canonical @ rotation.transpose(0, 1)
    return dense.to(torch.complex128), weights


def median_time(function, repeats: int) -> tuple[float, complex]:
    function()
    elapsed = []
    value = None
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = function()
        elapsed.append(time.perf_counter() - start)
    assert value is not None
    return statistics.median(elapsed), complex(value.detach().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="16,32,64,128")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--two-body-max-d", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/gate_a_agp_scaling.json"),
    )
    args = parser.parse_args()
    dimensions = [int(value) for value in args.dimensions.split(",")]
    records = []
    for dimension in dimensions:
        if dimension % 4:
            raise ValueError("benchmark dimensions must be divisible by four")
        pairs = dimension // 4
        channels = dimension // 2
        pair_matrix, canonical_weights = dense_canonical_pair_matrix(
            dimension, 200 + dimension
        )
        generator = torch.Generator().manual_seed(300 + dimension)
        raw = torch.randn(
            dimension, dimension, generator=generator, dtype=torch.float64
        )
        one_body = torch.diag(raw.diagonal()).to(torch.complex128)

        print(f"D={dimension} N={2*pairs} norm", flush=True)
        norm_time, norm_value = median_time(
            lambda: agp_norm_generating(pair_matrix, pairs), args.repeats
        )
        print(f"D={dimension} N={2*pairs} one_body", flush=True)
        one_body_time, one_body_value = median_time(
            lambda: agp_one_body_expectation(pair_matrix, pairs, one_body),
            args.repeats,
        )
        analytic_norm = float(
            elementary_symmetric(canonical_weights.square(), pairs).item()
        )
        timings = {
            "norm": norm_time,
            "one_body": one_body_time,
        }
        values = {
            "norm": [norm_value.real, norm_value.imag],
            "one_body": [one_body_value.real, one_body_value.imag],
        }
        if dimension <= args.two_body_max_d:
            left = torch.diag(torch.linspace(-0.3, 0.4, dimension)).to(
                torch.complex128
            )
            right = torch.diag(torch.linspace(0.2, 0.7, dimension)).to(
                torch.complex128
            )
            left_factors = torch.stack((left, right))
            right_factors = torch.stack((right, left))
            factor_weights = torch.tensor([0.2, -0.1], dtype=torch.complex128)
            print(f"D={dimension} N={2*pairs} two_body_L2", flush=True)
            two_body_time, two_body_value = median_time(
                lambda: agp_two_body_expectation_factorized(
                    pair_matrix,
                    pairs,
                    left_factors,
                    right_factors,
                    factor_weights,
                ),
                args.repeats,
            )
            timings["two_body_L2"] = two_body_time
            values["two_body_L2"] = [two_body_value.real, two_body_value.imag]

        counts = agp_structural_counts(dimension, pairs, channels)
        records.append(
            {
                "D": dimension,
                "N": 2 * pairs,
                "pairs": pairs,
                "channels": channels,
                "timing_seconds_median": timings,
                "values": values,
                "analytic_canonical_norm": analytic_norm,
                "norm_relative_error": abs(norm_value.real - analytic_norm)
                / max(abs(analytic_norm), 1.0),
                "structural_counts": {
                    key: str(value) if value > 2**53 else value
                    for key, value in counts.items()
                }
                | {
                    "dense_complex_pair_matrix_bytes": 16 * dimension**2,
                    "generating_recurrence_dense_matmuls": pairs + 1,
                    "generating_recurrence_asymptotic_memory_entries": dimension**2
                    + 2 * pairs,
                },
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "gate_a_fixed_number_pfaffian_scaling",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "complex128",
            "torch_threads": torch.get_num_threads(),
        },
        "scan": {
            "dimensions": dimensions,
            "repeats": args.repeats,
            "two_body_max_d": args.two_body_max_d,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
