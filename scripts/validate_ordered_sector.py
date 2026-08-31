"""Validate the ordered-coordinate harmonic-grid oracle against exterior truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from femps.hamiltonians import antisymmetric_many_body_hamiltonian
from femps.ordered_sector import (
    finite_difference_harmonic_hamiltonian,
    ordered_sector_hamiltonian,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=3)
    parser.add_argument("--spacing", type=float, default=0.6)
    parser.add_argument("--grid-points", type=int, nargs="+", default=[7, 9, 11, 13])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/ordered_sector_harmonic.json"),
    )
    args = parser.parse_args()
    continuum_energy = args.particles**2 / 2
    points = []
    for grid_points in args.grid_points:
        grid, one_body = finite_difference_harmonic_hamiltonian(
            grid_points, args.spacing
        )
        ordered = ordered_sector_hamiltonian(one_body, args.particles)
        exterior = antisymmetric_many_body_hamiltonian(one_body, args.particles)
        ordered_energy = float(torch.linalg.eigvalsh(ordered)[0])
        exterior_energy = float(torch.linalg.eigvalsh(exterior)[0])
        orbital_energy = float(
            torch.linalg.eigvalsh(one_body)[: args.particles].sum()
        )
        points.append(
            {
                "grid_points": grid_points,
                "spacing": args.spacing,
                "grid_min": float(grid[0]),
                "grid_max": float(grid[-1]),
                "ordered_dimension": ordered.shape[0],
                "matrix_max_abs_difference": float(
                    torch.max(torch.abs(ordered - exterior))
                ),
                "ordered_ground_energy": ordered_energy,
                "exterior_ground_energy": exterior_energy,
                "occupied_orbital_energy_sum": orbital_energy,
                "ordered_exterior_absolute_difference": abs(
                    ordered_energy - exterior_energy
                ),
                "ordered_orbital_absolute_difference": abs(
                    ordered_energy - orbital_energy
                ),
                "error_vs_continuum": ordered_energy - continuum_energy,
            }
        )
    result = {
        "schema_version": 1,
        "experiment": "ordered_sector_harmonic_grid_oracle",
        "particles": args.particles,
        "continuum_energy": continuum_energy,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    worst = max(point["matrix_max_abs_difference"] for point in points)
    print(
        f"N={args.particles} points={len(points)} "
        f"max_ordered_exterior_matrix_error={worst:.3e} "
        f"finest_energy={points[-1]['ordered_ground_energy']:.12f}"
    )


if __name__ == "__main__":
    main()
