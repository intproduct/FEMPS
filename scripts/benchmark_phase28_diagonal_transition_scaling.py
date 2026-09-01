"""Measure the Phase 28 diagonal-transition cost and inverse-path speedup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from femps.algorithms import canonical_slater_orbitals
from femps.benchmarks import ProcessRSSMonitor
from femps.exterior import (
    diagonal_path_hamiltonian_matrices,
    diagonal_path_structural_counts,
    diagonal_path_transition_diagnostics,
)
from femps.hamiltonians import harmonic_pair_hamiltonian


def _orbitals(terms: int, dimension: int, particles: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(
        (terms, dimension, particles), generator=generator, dtype=torch.float64
    )
    imaginary = torch.randn(
        (terms, dimension, particles), generator=generator, dtype=torch.float64
    )
    return canonical_slater_orbitals(torch.complex(real, imaginary))


def _operators(
    dimension: int, factor_rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    one_body, interaction = harmonic_pair_hamiltonian(
        dimension, kappa=0.35, dtype=torch.complex128, device="cpu"
    )
    left = interaction.left[:1].repeat(factor_rank, 1, 1)
    right = interaction.right[:1].repeat(factor_rank, 1, 1)
    weights = interaction.weights[:1].repeat(factor_rank) / factor_rank
    return one_body, left, right, weights


def _evaluate(
    orbitals: torch.Tensor,
    one_body: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    weights: torch.Tensor,
    *,
    algorithm: str,
    backward: bool,
) -> torch.Tensor:
    state = orbitals.detach().clone().requires_grad_(backward)
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        state,
        one_body,
        two_body_left=left,
        two_body_right=right,
        two_body_weights=weights,
        transition_algorithm=algorithm,
    )
    value = overlap.real.square().sum() + hamiltonian.real.square().sum()
    if backward:
        value.backward()
        assert state.grad is not None
    return hamiltonian.detach()


def _timings(
    orbitals: torch.Tensor,
    operators: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    algorithm: str,
    backward: bool,
    repeats: int,
) -> tuple[list[float], dict]:
    _evaluate(
        orbitals, *operators, algorithm=algorithm, backward=backward
    )
    samples = []
    with ProcessRSSMonitor() as monitor:
        for _ in range(repeats):
            started = time.perf_counter()
            _evaluate(
                orbitals, *operators, algorithm=algorithm, backward=backward
            )
            samples.append(time.perf_counter() - started)
    return samples, monitor.record().as_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase28_diagonal_transition_scaling.json"
        ),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    specifications = [
        (2, 8, 4, 1, "N"),
        (4, 8, 4, 1, "baseline"),
        (6, 8, 4, 1, "N"),
        (4, 6, 4, 1, "D"),
        (4, 10, 4, 1, "D"),
        (4, 8, 1, 1, "K"),
        (4, 8, 2, 1, "K"),
        (4, 8, 8, 1, "K"),
        (4, 8, 4, 2, "L"),
        (4, 8, 4, 4, "L"),
    ]
    points = []
    for index, (particles, dimension, terms, factor_rank, axis) in enumerate(
        specifications
    ):
        orbitals = _orbitals(terms, dimension, particles, seed=2800 + index)
        operators = _operators(dimension, factor_rank)
        auto_hamiltonian = _evaluate(
            orbitals, *operators, algorithm="auto", backward=False
        )
        minor_hamiltonian = _evaluate(
            orbitals, *operators, algorithm="minor", backward=False
        )
        absolute_difference = float(
            torch.max(torch.abs(auto_hamiltonian - minor_hamiltonian))
        )
        modes = {}
        for algorithm in ("auto", "minor"):
            for backward in (False, True):
                samples, memory = _timings(
                    orbitals,
                    operators,
                    algorithm=algorithm,
                    backward=backward,
                    repeats=args.repeats,
                )
                label = f"{algorithm}_{'forward_backward' if backward else 'forward'}"
                modes[label] = {
                    "samples_seconds": samples,
                    "median_seconds": statistics.median(samples),
                    "minimum_seconds": min(samples),
                    "cpu_memory": memory,
                }
        counts = diagonal_path_structural_counts(
            particles, dimension, terms, factor_rank
        )
        point = {
            "axis": axis,
            "N": particles,
            "D": dimension,
            "K": terms,
            "L": factor_rank,
            "auto_minor_max_absolute_difference": absolute_difference,
            "transition_diagnostics": diagonal_path_transition_diagnostics(
                orbitals
            ),
            "structural_counts": counts,
            "modes": modes,
            "forward_speedup_minor_over_auto": (
                modes["minor_forward"]["median_seconds"]
                / modes["auto_forward"]["median_seconds"]
            ),
            "forward_backward_speedup_minor_over_auto": (
                modes["minor_forward_backward"]["median_seconds"]
                / modes["auto_forward_backward"]["median_seconds"]
            ),
        }
        points.append(point)
        print(
            f"N={particles} D={dimension} K={terms} L={factor_rank}: "
            f"forward speedup={point['forward_speedup_minor_over_auto']:.2f}x "
            f"fb speedup={point['forward_backward_speedup_minor_over_auto']:.2f}x",
            flush=True,
        )
    artifact = {
        "schema_version": 1,
        "experiment": "phase28_diagonal_transition_scaling",
        "evidence_level": "numerical",
        "device": "cpu",
        "repeats": args.repeats,
        "algorithm_boundary": (
            "inverse determinant derivatives for well-conditioned overlaps; "
            "automatic singular-safe minor fallback"
        ),
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "points": len(points)}, indent=2))


if __name__ == "__main__":
    main()
