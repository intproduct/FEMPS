"""E5a: six noninteracting spinless harmonic fermions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from femps.algorithms import FiniteAgpConfig, run_finite_agp_variable_projection
from femps.basis import harmonic_hamiltonian
from femps.exterior import (
    agp_exterior_coefficients,
    agp_femps_cores,
    agp_tensor,
    antisymmetry_residual,
    best_rank_error,
    femps_exterior_coefficients,
    materialize_femps_paths,
    normalized_slater_from_minors,
    particle_schmidt_spectrum,
    particle_tt_ranks,
    slater_flat_spectrum,
    slater_sum_cores,
)
from femps.hamiltonians import (
    agp_energy,
    antisymmetric_many_body_hamiltonian,
    exact_noninteracting_fermion_energy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=8)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/fermion_e5a.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("docs/experiments/results/fermion_e5a_checkpoint.pt"),
    )
    args = parser.parse_args()
    particles = 6
    pairs = 3
    if args.basis_order < particles:
        raise ValueError("six fermions require basis_order >= 6")
    orbitals = torch.eye(args.basis_order, dtype=torch.complex128)[:, :particles]
    slater = normalized_slater_from_minors(orbitals)
    chi_one_state = materialize_femps_paths(
        slater_sum_cores(orbitals.unsqueeze(0))
    )
    left = orbitals[:, (0, 2, 4)].transpose(0, 1)
    right = orbitals[:, (1, 3, 5)].transpose(0, 1)
    pair_matrix = left.transpose(0, 1) @ right - right.transpose(0, 1) @ left
    pfaffian_state = agp_tensor(pair_matrix, pairs)
    ordered_coefficients = femps_exterior_coefficients(
        agp_femps_cores(left, right, pairs=pairs)
    )
    pfaffian_coefficients = agp_exterior_coefficients(pair_matrix, pairs)
    one_body = harmonic_hamiltonian(args.basis_order, dtype=torch.complex128)
    constructed_energy = float(agp_energy(pair_matrix, pairs, one_body))
    finite_truth = float(
        torch.linalg.eigvalsh(
            antisymmetric_many_body_hamiltonian(one_body, particles)
        )[0].real
    )
    expected_energy = exact_noninteracting_fermion_energy(particles)

    spectra = []
    for cut in range(1, particles):
        observed = particle_schmidt_spectrum(slater, cut)
        expected = slater_flat_spectrum(particles, cut, dtype=observed.dtype)
        spectra.append(
            {
                "cut": cut,
                "multiplicity": expected.numel(),
                "expected_singular_value": float(expected[0]),
                "max_abs_error": float(
                    torch.max(torch.abs(observed[: expected.numel()] - expected))
                ),
            }
        )
    central_spectrum = particle_schmidt_spectrum(slater, particles // 2)
    central_truncation = [
        {
            "rank": rank,
            "observed_relative_error": float(
                best_rank_error(central_spectrum, rank)
            ),
            "closed_form_relative_error": math.sqrt((20 - rank) / 20),
        }
        for rank in (1, 5, 10, 15, 19)
    ]
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
    result = {
        "schema_version": 1,
        "experiment": "fermion_e5a_six_noninteracting_fermions",
        "D": args.basis_order,
        "N": particles,
        "constructed": {
            "continuum_reference_energy": expected_energy,
            "finite_basis_reference_energy": finite_truth,
            "energy": constructed_energy,
            "energy_absolute_error": abs(constructed_energy - expected_energy),
            "chi_one_femps_max_abs_error": float(
                torch.max(torch.abs(chi_one_state - slater))
            ),
            "ordered_pfaffian_coefficient_max_abs_error": float(
                torch.max(
                    torch.abs(ordered_coefficients - pfaffian_coefficients)
                )
            ),
            "pfaffian_tensor_max_abs_error": float(
                torch.max(torch.abs(pfaffian_state - slater))
            ),
            "antisymmetry_residual": float(antisymmetry_residual(slater)),
            "ordinary_particle_tt_ranks": list(particle_tt_ranks(slater)),
            "femps_correlation_bonds": [1] * (particles - 1),
            "pfaffian_pair_channels": pairs,
            "schmidt_spectra": spectra,
            "central_cut_truncation": central_truncation,
        },
        "blind_training": training,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"constructed_E={constructed_energy:.12f} "
        f"blind_E={training['final_energy']:.12f} "
        f"blind_error={training['error_vs_finite_basis']:.3e} "
        f"ranks={result['constructed']['ordinary_particle_tt_ranks']}"
    )


if __name__ == "__main__":
    main()
