"""Stress-test scaled AGP overlaps and the positive stable norm recurrence."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from femps.exterior import agp_log_norm, agp_norm_generating, agp_overlap_generating


def _legacy_newton_norm(pair_matrix: torch.Tensor, pairs: int) -> torch.Tensor:
    overlap_matrix = pair_matrix.conj().transpose(0, 1) @ pair_matrix
    power_matrix = torch.eye(
        pair_matrix.shape[0], dtype=pair_matrix.dtype, device=pair_matrix.device
    )
    traces = [torch.zeros((), dtype=pair_matrix.dtype, device=pair_matrix.device)]
    for _ in range(pairs):
        power_matrix = power_matrix @ overlap_matrix
        traces.append(torch.trace(power_matrix))
    coefficients = [
        torch.ones((), dtype=pair_matrix.dtype, device=pair_matrix.device)
    ]
    for degree in range(1, pairs + 1):
        terms = []
        for power in range(1, degree + 1):
            sign = -1 if (power + 1) % 2 else 1
            terms.append(
                0.5 * sign * traces[power] * coefficients[degree - power]
            )
        coefficients.append(torch.stack(terms).sum() / degree)
    return coefficients[pairs].real


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/agp_stability_stress.json"),
    )
    args = parser.parse_args()
    if args.dimension < 2 or args.dimension % 2:
        raise ValueError("dimension must be positive and even")

    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    raw = torch.randn(args.dimension, args.dimension, generator=generator, dtype=torch.float64)
    pair_matrix = (raw - raw.transpose(0, 1)).to(device)
    pairs = args.dimension // 2
    _, reference_log_norm = torch.linalg.slogdet(pair_matrix)
    reference_norm = torch.exp(reference_log_norm)

    started = time.perf_counter()
    legacy_norm = _legacy_newton_norm(pair_matrix, pairs)
    legacy_seconds = time.perf_counter() - started
    started = time.perf_counter()
    stable_norm = agp_norm_generating(pair_matrix, pairs)
    stable_seconds = time.perf_counter() - started

    scale_records = []
    for scale in (1e-12, 1e-6, 1.0, 1e6, 1e12):
        observed_log_norm = agp_log_norm(scale * pair_matrix, pairs)
        expected_log_norm = reference_log_norm + 2 * pairs * math.log(scale)
        observed_norm = agp_norm_generating(scale * pair_matrix, pairs)
        scale_records.append(
            {
                "pair_scale": scale,
                "log_norm": float(observed_log_norm.detach().cpu()),
                "log_norm_absolute_error": float(
                    torch.abs(observed_log_norm - expected_log_norm).detach().cpu()
                ),
                "ordinary_norm_is_finite": bool(torch.isfinite(observed_norm).detach().cpu()),
                "ordinary_norm_is_nonzero": bool((observed_norm != 0).detach().cpu()),
            }
        )

    second_raw = torch.randn(
        args.dimension,
        args.dimension,
        generator=generator,
        dtype=torch.float64,
    )
    second_pair = (second_raw - second_raw.transpose(0, 1)).to(device)
    transition_pairs = min(8, pairs)
    transition_reference = agp_overlap_generating(
        pair_matrix, second_pair, transition_pairs
    )
    transition_scaled = agp_overlap_generating(
        1e-120 * pair_matrix, 1e120 * second_pair, transition_pairs
    )

    record = {
        "schema_version": 1,
        "experiment": "agp_stability_stress",
        "dimension": args.dimension,
        "particles": 2 * pairs,
        "pairs": pairs,
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "reference_log_norm": float(reference_log_norm.detach().cpu()),
        "legacy_newton_norm": float(legacy_norm.detach().cpu()),
        "legacy_relative_error": float(
            (torch.abs(legacy_norm - reference_norm) / reference_norm).detach().cpu()
        ),
        "legacy_seconds": legacy_seconds,
        "stable_norm": float(stable_norm.detach().cpu()),
        "stable_relative_error": float(
            (torch.abs(stable_norm - reference_norm) / reference_norm).detach().cpu()
        ),
        "stable_seconds": stable_seconds,
        "scale_stress": scale_records,
        "reciprocal_transition_scale": 1e120,
        "transition_pairs": transition_pairs,
        "scaled_transition_relative_error": float(
            (
                torch.abs(transition_scaled - transition_reference)
                / torch.abs(transition_reference)
            )
            .detach()
            .cpu()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
