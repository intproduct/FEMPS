"""Phase 8: non-materialized eight-fermion Pfaffian/FEMPS baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from femps.algorithms import FiniteAgpConfig, run_finite_agp_variable_projection
from femps.basis import harmonic_hamiltonian
from femps.exterior import agp_exterior_coefficients
from femps.hamiltonians import (
    agp_energy,
    antisymmetric_many_body_hamiltonian,
    exact_noninteracting_fermion_energy,
)


def occupied_pair_matrix(
    basis_order: int,
    particles: int,
    *,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Return the canonical AGP pairing consecutive occupied orbitals."""

    if particles % 2:
        raise ValueError("this benchmark requires an even particle count")
    if basis_order < particles:
        raise ValueError("basis_order must be at least particles")
    matrix = torch.zeros((basis_order, basis_order), dtype=dtype)
    for orbital in range(0, particles, 2):
        matrix[orbital, orbital + 1] = 1
        matrix[orbital + 1, orbital] = -1
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=10)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/fermion_e8a.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("docs/experiments/results/fermion_e8a_checkpoint.pt"),
    )
    args = parser.parse_args()

    particles = 8
    pairs = particles // 2
    if args.basis_order**particles <= 2_000_000:
        raise ValueError("E8a must exercise the non-materialized particle-tensor path")

    pair_matrix = occupied_pair_matrix(args.basis_order, particles)
    one_body = harmonic_hamiltonian(args.basis_order, dtype=torch.complex128)
    exterior_hamiltonian = antisymmetric_many_body_hamiltonian(one_body, particles)
    exterior_values, exterior_vectors = torch.linalg.eigh(exterior_hamiltonian)
    exterior_coefficients = agp_exterior_coefficients(pair_matrix, pairs)
    exterior_norm = torch.vdot(exterior_coefficients, exterior_coefficients).real
    exterior_energy = float(
        (
            torch.vdot(
                exterior_coefficients,
                exterior_hamiltonian @ exterior_coefficients,
            )
            / exterior_norm
        ).real
    )
    polynomial_energy = float(agp_energy(pair_matrix, pairs, one_body))
    expected_energy = exact_noninteracting_fermion_energy(particles)
    fidelity = float(
        (
            torch.abs(torch.vdot(exterior_vectors[:, 0], exterior_coefficients)) ** 2
            / exterior_norm
        ).real
    )

    config = FiniteAgpConfig(
        basis_order=args.basis_order,
        particles=particles,
        agp_terms=1,
        kappa=0.0,
        steps=args.steps,
        learning_rate=2e-2,
        final_learning_rate=1e-5,
        seed=args.seed,
        device=args.device,
        record_points=20,
        checkpoint_every=100,
    )
    training = run_finite_agp_variable_projection(
        config,
        checkpoint_path=args.checkpoint,
        resume=args.resume and args.checkpoint.exists(),
    )
    if training["ordinary_particle_tt_ranks"] is not None:
        raise RuntimeError("training unexpectedly materialized the ordinary tensor")

    structural_ranks = [1] + [math.comb(particles, cut) for cut in range(1, particles)] + [1]
    result = {
        "schema_version": 1,
        "experiment": "fermion_e8a_eight_noninteracting_fermions",
        "D": args.basis_order,
        "N": particles,
        "materialization": {
            "ordinary_particle_tensor_materialized": False,
            "ordinary_particle_tensor_entries": args.basis_order**particles,
            "exterior_sector_materialized": True,
            "exterior_sector_dimension": math.comb(args.basis_order, particles),
        },
        "constructed": {
            "continuum_reference_energy": expected_energy,
            "finite_basis_reference_energy": float(exterior_values[0].real),
            "polynomial_energy": polynomial_energy,
            "exterior_energy": exterior_energy,
            "polynomial_exterior_absolute_difference": abs(
                polynomial_energy - exterior_energy
            ),
            "energy_absolute_error": abs(polynomial_energy - expected_energy),
            "finite_basis_ground_fidelity": fidelity,
            "exterior_coefficient_norm": float(exterior_norm),
            "ordinary_particle_tt_ranks_from_slater_theorem": structural_ranks,
            "central_rank_69_best_relative_error": math.sqrt(1 / 70),
            "femps_correlation_bonds": [1] * (particles - 1),
            "pfaffian_pair_channels": pairs,
        },
        "blind_training": training,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"constructed_E={polynomial_energy:.12f} "
        f"blind_E={training['final_energy']:.12f} "
        f"blind_error={training['error_vs_finite_basis']:.3e} "
        f"particle_entries={args.basis_order**particles} "
        f"exterior_dimension={math.comb(args.basis_order, particles)}"
    )


if __name__ == "__main__":
    main()
