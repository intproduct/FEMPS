"""Compare ordered-sector, exterior-CI, and ordinary particle-TT diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from femps.exterior import antisymmetry_residual, particle_tt_ranks
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian_dense_two_body,
)
from femps.ordered_sector import (
    extend_from_ordered_sector,
    finite_difference_harmonic_hamiltonian,
    ordered_sector_hamiltonian,
)


def _point(particles: int, grid_points: int, spacing: float) -> dict:
    grid, one_body = finite_difference_harmonic_hamiltonian(grid_points, spacing)
    pair_potential = 1 / torch.sqrt(
        (grid[:, None] - grid[None, :]) ** 2 + 1
    )
    started = time.perf_counter()
    ordered = ordered_sector_hamiltonian(
        one_body, particles, pair_potential=pair_potential
    )
    ordered_build_seconds = time.perf_counter() - started
    identity = torch.eye(grid_points, dtype=one_body.dtype)
    dense_two_body = torch.einsum(
        "pq,pr,qs->pqrs", pair_potential, identity, identity
    )
    started = time.perf_counter()
    exterior = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, particles, dense_two_body
    )
    exterior_build_seconds = time.perf_counter() - started
    values, vectors = torch.linalg.eigh(ordered)
    particle_state = extend_from_ordered_sector(
        vectors[:, 0], grid_points, particles
    )
    return {
        "N": particles,
        "grid_points": grid_points,
        "spacing": spacing,
        "grid_min": float(grid[0]),
        "grid_max": float(grid[-1]),
        "ordered_dimension": ordered.shape[0],
        "ordered_build_seconds": ordered_build_seconds,
        "exterior_build_seconds": exterior_build_seconds,
        "matrix_max_absolute_difference": float(
            torch.max(torch.abs(ordered - exterior))
        ),
        "ground_energy": float(values[0]),
        "ordinary_particle_tt_ranks": list(particle_tt_ranks(particle_state)),
        "antisymmetry_residual": float(antisymmetry_residual(particle_state)),
        "explicit_particle_tensor_elements": particle_state.numel(),
        "ordered_vector_elements": vectors.shape[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--grid-points", nargs="+", type=int, default=[8, 10, 12])
    parser.add_argument("--spacing", type=float, default=0.7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_ordered_sector_comparison.json"
        ),
    )
    args = parser.parse_args()
    points = [
        _point(args.particles, grid_points, args.spacing)
        for grid_points in args.grid_points
    ]
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_ordered_sector_exterior_tt_comparison",
        "evidence_level": "numerical_exact_at_fixed_grid",
        "ordered_sector_is_an_exponential_truth_oracle_not_a_production_TN": True,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"points={len(points)} worst_matrix_error="
        f"{max(point['matrix_max_absolute_difference'] for point in points):.3e} "
        f"largest_TT_ranks={points[-1]['ordinary_particle_tt_ranks']}"
    )


if __name__ == "__main__":
    main()
