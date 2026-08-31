"""Build the deterministic Phase 14 ordered-sector latticeTN comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from femps.baselines.ordered_functional_mps import (
    ordered_sector_dense_energy_from_mps,
    ordered_sector_functional_mps,
    ordered_values_from_mps,
    ordered_values_to_particle_tensor,
)
from femps.exterior import antisymmetry_residual, particle_tt_ranks
from femps.ordered_sector import (
    extend_from_ordered_sector,
    finite_difference_harmonic_hamiltonian,
    ordered_sector_hamiltonian,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=0.7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase14_ordered_latticetn_comparator.json"
        ),
    )
    arguments = parser.parse_args()
    grid, one_body = finite_difference_harmonic_hamiltonian(
        arguments.dimension, arguments.spacing
    )
    pair_potential = 1 / torch.sqrt(
        (grid[:, None] - grid[None, :]) ** 2 + 1
    )
    hamiltonian = ordered_sector_hamiltonian(
        one_body, arguments.particles, pair_potential=pair_potential
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
    values = eigenvectors[:, 0]
    mps, ranks, discarded_norm = ordered_sector_functional_mps(
        values, arguments.dimension, arguments.particles
    )
    ordered_tensor = ordered_values_to_particle_tensor(
        values, arguments.dimension, arguments.particles
    )
    antisymmetric_tensor = extend_from_ordered_sector(
        values, arguments.dimension, arguments.particles
    )
    energy = ordered_sector_dense_energy_from_mps(mps, hamiltonian)
    energy.backward()
    gradient_norms = [float(torch.linalg.vector_norm(core.grad)) for core in mps.tensors]
    reconstructed = ordered_values_from_mps(mps)
    result = {
        "schema_version": 1,
        "experiment": "phase14_ordered_sector_latticetn_comparator",
        "evidence_level": "numerical_exact_at_fixed_grid",
        "production_solver": False,
        "particles": arguments.particles,
        "dimension": arguments.dimension,
        "spacing": arguments.spacing,
        "ground_energy": float(eigenvalues[0]),
        "mps_energy": float(energy.detach()),
        "energy_absolute_error": float(
            torch.abs(energy.detach() - eigenvalues[0])
        ),
        "ordered_dimension": values.numel(),
        "ordered_particle_tensor_elements": ordered_tensor.numel(),
        "ordered_mps_ranks": list(ranks),
        "ordered_mps_parameters": sum(core.numel() for core in mps.tensors),
        "antisymmetric_particle_tt_ranks": list(
            particle_tt_ranks(antisymmetric_tensor)
        ),
        "ordered_particle_tt_ranks": list(particle_tt_ranks(ordered_tensor)),
        "native_norm_absolute_error": float(
            torch.abs(mps.norm_sq().detach() - torch.vdot(values, values))
        ),
        "ordered_reconstruction_max_absolute_error": float(
            torch.max(torch.abs(reconstructed.detach() - values))
        ),
        "tt_svd_discarded_norm": float(discarded_norm),
        "antisymmetry_residual_after_signed_extension": float(
            antisymmetry_residual(antisymmetric_tensor)
        ),
        "all_ad_gradients_finite": all(
            bool(torch.isfinite(core.grad).all()) for core in mps.tensors
        ),
        "ad_gradient_norms_at_ground_state": gradient_norms,
        "limitations": [
            "ordered energy currently gathers a dense D**N tensor",
            "hard ordering constraint is not yet a scalable MPO",
            "this comparator is not named FEMPS",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"N={arguments.particles} D={arguments.dimension} "
        f"ordered_ranks={ranks} "
        f"antisymmetric_ranks={tuple(result['antisymmetric_particle_tt_ranks'])} "
        f"energy_error={result['energy_absolute_error']:.3e}"
    )


if __name__ == "__main__":
    main()
