"""Deterministic Gate C controls for the ordered-distance native MPS/MPO path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from femps.baselines.ordered_distance_mpo import (
    compress_mpo,
    gap_charge_projector_mpo,
    gap_soft_coulomb_hamiltonian_mpo,
    ordered_values_to_gap_mps,
)
from femps.ordered_distance import (
    gap_configurations,
    gap_hamiltonian,
    gap_values_to_ordered_values,
)


def _max_mpo_bond(mpo) -> int:
    return max(max(tensor.shape[:2]) for tensor in mpo.tensors)


def _flat_gap_indices(
    grid_points: int, particles: int, gap_cutoff: int
) -> torch.Tensor:
    sites = particles + 1
    local_dimension = gap_cutoff + 1
    return torch.tensor(
        [
            sum(
                value * local_dimension ** (sites - 1 - site)
                for site, value in enumerate(gaps)
            )
            for gaps in gap_configurations(
                grid_points, particles, gap_cutoff=gap_cutoff
            )
        ],
        dtype=torch.long,
    )


def _ground_state(
    grid_points: int, particles: int, spacing: float, gap_cutoff: int
):
    hamiltonian = gap_hamiltonian(
        grid_points,
        particles,
        spacing,
        gap_cutoff=gap_cutoff,
        soft_coulomb=True,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
    ordered_values = gap_values_to_ordered_values(
        eigenvectors[:, 0],
        grid_points,
        particles,
        gap_cutoff=gap_cutoff,
    )
    return hamiltonian, eigenvalues[0], ordered_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--grid-points", type=int, default=8)
    parser.add_argument("--spacing", type=float, default=0.7)
    parser.add_argument("--gap-cutoffs", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--mps-bonds", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--mpo-bonds", nargs="+", type=int, default=[4, 8, 16, 24, 32, 64])
    parser.add_argument("--grid-sweep", nargs="+", type=int, default=[6, 8, 10, 12])
    parser.add_argument(
        "--fixed-box-cases",
        nargs="+",
        default=["8:0.9", "10:0.7", "13:0.525", "15:0.45"],
        help="grid_points:spacing pairs with a common intended half-box",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase15_ordered_distance_gate.json"
        ),
    )
    arguments = parser.parse_args()
    particles = arguments.particles
    grid_points = arguments.grid_points
    spacing = arguments.spacing
    holes = grid_points - particles
    if holes < 0:
        raise ValueError("grid_points must be at least particles")

    full_hamiltonian, full_energy, full_ordered_values = _ground_state(
        grid_points, particles, spacing, holes
    )
    cutoff_points = []
    for cutoff in arguments.gap_cutoffs:
        started = time.perf_counter()
        hamiltonian, energy, ordered_values = _ground_state(
            grid_points, particles, spacing, cutoff
        )
        mps, ranks, discarded = ordered_values_to_gap_mps(
            ordered_values,
            grid_points,
            particles,
            gap_cutoff=cutoff,
            requires_grad=False,
        )
        mpo = gap_soft_coulomb_hamiltonian_mpo(
            grid_points, particles, spacing, gap_cutoff=cutoff
        )
        native_energy = mps.energy_with_MPO(mpo)
        cutoff_points.append(
            {
                "gap_cutoff": cutoff,
                "local_dimension": cutoff + 1,
                "sector_dimension": hamiltonian.shape[0],
                "ground_energy": float(energy),
                "error_vs_full_gap_basis": float(energy - full_energy),
                "native_energy_absolute_error": float(
                    torch.abs(native_energy - energy)
                ),
                "exact_mps_ranks": list(ranks),
                "exact_mps_parameters": sum(core.numel() for core in mps.tensors),
                "tt_svd_discarded_norm": float(discarded),
                "raw_mpo_max_bond": _max_mpo_bond(mpo),
                "seconds": time.perf_counter() - started,
            }
        )

    full_mpo = gap_soft_coulomb_hamiltonian_mpo(
        grid_points, particles, spacing
    )
    full_mps, full_ranks, full_discarded = ordered_values_to_gap_mps(
        full_ordered_values,
        grid_points,
        particles,
        requires_grad=False,
    )
    charge_projector = gap_charge_projector_mpo(grid_points, particles)
    mps_points = []
    for bond in arguments.mps_bonds:
        mps, ranks, discarded = ordered_values_to_gap_mps(
            full_ordered_values,
            grid_points,
            particles,
            max_bond=bond,
            requires_grad=False,
        )
        energy = mps.energy_with_MPO(full_mpo)
        charge_weight = (
            mps._expect_MPO(charge_projector) / mps.overlap(mps)
        ).real
        mps_points.append(
            {
                "max_bond": bond,
                "retained_ranks": list(ranks),
                "parameters": sum(core.numel() for core in mps.tensors),
                "discarded_norm": float(discarded),
                "charge_weight": float(charge_weight),
                "energy": float(energy),
                "energy_error_vs_full": float(energy - full_energy),
            }
        )

    mpo_points = []
    for bond in arguments.mpo_bonds:
        compressed, ranks, discarded = compress_mpo(full_mpo, bond)
        energy = full_mps.energy_with_MPO(compressed)
        mpo_points.append(
            {
                "max_bond": bond,
                "retained_ranks": list(ranks),
                "local_discarded_singular_norm": float(discarded),
                "energy": float(energy),
                "energy_error_on_exact_ground_state": float(energy - full_energy),
            }
        )

    operator_cutoff = min(2, holes)
    operator_exact = gap_soft_coulomb_hamiltonian_mpo(
        grid_points,
        particles,
        spacing,
        gap_cutoff=operator_cutoff,
    )
    operator_indices = _flat_gap_indices(
        grid_points, particles, operator_cutoff
    )
    operator_matrix = operator_exact.to_dense()[operator_indices][:, operator_indices]
    operator_norm = torch.linalg.matrix_norm(operator_matrix)
    operator_points = []
    for bond in arguments.mpo_bonds:
        compressed, ranks, discarded = compress_mpo(operator_exact, bond)
        matrix = compressed.to_dense()[operator_indices][:, operator_indices]
        difference = matrix - operator_matrix
        operator_points.append(
            {
                "max_bond": bond,
                "retained_ranks": list(ranks),
                "local_discarded_singular_norm": float(discarded),
                "frobenius_operator_error": float(torch.linalg.matrix_norm(difference)),
                "relative_frobenius_operator_error": float(
                    torch.linalg.matrix_norm(difference) / operator_norm
                ),
            }
        )

    grid_points_results = []
    for current_grid in arguments.grid_sweep:
        current_holes = current_grid - particles
        if current_holes < 0:
            continue
        started = time.perf_counter()
        hamiltonian, energy, ordered_values = _ground_state(
            current_grid, particles, spacing, current_holes
        )
        mps, ranks, _ = ordered_values_to_gap_mps(
            ordered_values,
            current_grid,
            particles,
            requires_grad=False,
        )
        mpo = gap_soft_coulomb_hamiltonian_mpo(
            current_grid, particles, spacing
        )
        native_energy = mps.energy_with_MPO(mpo)
        grid_points_results.append(
            {
                "grid_points": current_grid,
                "box_min": -spacing * (current_grid - 1) / 2,
                "box_max": spacing * (current_grid - 1) / 2,
                "sector_dimension": hamiltonian.shape[0],
                "local_gap_dimension": current_holes + 1,
                "ground_energy": float(energy),
                "native_energy_absolute_error": float(
                    torch.abs(native_energy - energy)
                ),
                "exact_mps_ranks": list(ranks),
                "raw_mpo_max_bond": _max_mpo_bond(mpo),
                "seconds": time.perf_counter() - started,
            }
        )

    fixed_box_results = []
    for specification in arguments.fixed_box_cases:
        current_grid_text, current_spacing_text = specification.split(":", maxsplit=1)
        current_grid = int(current_grid_text)
        current_spacing = float(current_spacing_text)
        current_holes = current_grid - particles
        started = time.perf_counter()
        hamiltonian, energy, _ = _ground_state(
            current_grid, particles, current_spacing, current_holes
        )
        fixed_box_results.append(
            {
                "grid_points": current_grid,
                "spacing": current_spacing,
                "box_min": -current_spacing * (current_grid - 1) / 2,
                "box_max": current_spacing * (current_grid - 1) / 2,
                "sector_dimension": hamiltonian.shape[0],
                "ground_energy": float(energy),
                "seconds": time.perf_counter() - started,
            }
        )

    ad_mps, _, _ = ordered_values_to_gap_mps(
        full_ordered_values,
        grid_points,
        particles,
        requires_grad=True,
    )
    ad_energy = ad_mps.energy_with_MPO(full_mpo)
    ad_energy.backward()
    result = {
        "schema_version": 1,
        "experiment": "phase15_ordered_distance_gate",
        "evidence_level": "exact_fixed_grid_plus_controlled_mps_mpo_truncations",
        "particles": particles,
        "grid_points": grid_points,
        "spacing": spacing,
        "full_gap_cutoff": holes,
        "full_ground_energy": float(full_energy),
        "full_sector_dimension": full_hamiltonian.shape[0],
        "full_exact_mps_ranks": list(full_ranks),
        "full_exact_mps_discarded_norm": float(full_discarded),
        "full_raw_mpo_max_bond": _max_mpo_bond(full_mpo),
        "gap_cutoff_sweep": cutoff_points,
        "mps_bond_sweep": mps_points,
        "mpo_bond_energy_sweep": mpo_points,
        "mpo_operator_error_audit": {
            "gap_cutoff": operator_cutoff,
            "sector_dimension": operator_matrix.shape[0],
            "points": operator_points,
        },
        "grid_sweep": grid_points_results,
        "fixed_box_spacing_sweep": fixed_box_results,
        "native_ad": {
            "energy_absolute_error": float(
                torch.abs(ad_energy.detach() - full_energy)
            ),
            "all_gradients_finite": all(
                bool(torch.isfinite(core.grad).all()) for core in ad_mps.tensors
            ),
            "gradient_norms_at_exact_ground_state": [
                float(torch.linalg.vector_norm(core.grad))
                for core in ad_mps.tensors
            ],
        },
        "production_path_materializes_D_power_N": False,
        "truth_operator_audit_materializes_small_local_gap_tensor": True,
        "complexity_controls": [
            "grid_points/box size",
            "gap_cutoff/local distance dimension",
            "MPS bond",
            "MPO compression bond",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"N={particles} L={grid_points} E={float(full_energy):.12f} "
        f"gap_error(q={arguments.gap_cutoffs[-2]})="
        f"{cutoff_points[-2]['error_vs_full_gap_basis']:.3e} "
        f"mps_error(chi={arguments.mps_bonds[-2]})="
        f"{mps_points[-2]['energy_error_vs_full']:.3e} "
        f"mpo_error(W={arguments.mpo_bonds[-2]})="
        f"{mpo_points[-2]['energy_error_on_exact_ground_state']:.3e}"
    )


if __name__ == "__main__":
    main()
