"""Run a blind single-AGP soft-Coulomb benchmark with normalized records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from femps.algorithms import (
    FiniteAgpConfig,
    canonical_pair_matrices,
    contribution_gram_spectrum,
    run_finite_agp_variable_projection,
    solve_generalized_hermitian,
)
from femps.basis import harmonic_hamiltonian
from femps.benchmarks import soft_coulomb_point_from_training
from femps.hamiltonians import (
    agp_hamiltonian_transition_matrices,
    soft_coulomb_operator,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, required=True)
    parser.add_argument("--basis-order", type=int, required=True)
    parser.add_argument("--quadrature-order", type=int, default=128)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-id", required=True)
    parser.add_argument("--largest-basis-reference-energy", type=float)
    parser.add_argument("--direct-dense-same-basis-reference-energy", type=float)
    parser.add_argument("--operator-error-estimate", type=float)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = FiniteAgpConfig(
        basis_order=args.basis_order,
        particles=args.particles,
        agp_terms=1,
        steps=args.steps,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        seed=args.seed,
        device=args.device,
        record_points=20,
        checkpoint_every=100,
        interaction_model="soft_coulomb",
        soft_coulomb_quadrature_order=args.quadrature_order,
        soft_coulomb_relative_threshold=1e-14,
    )
    training = run_finite_agp_variable_projection(
        config, checkpoint_path=args.checkpoint
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pairs = canonical_pair_matrices(payload["best_raw"])
    one_body = harmonic_hamiltonian(args.basis_order, dtype=torch.complex128)
    interaction, _ = soft_coulomb_operator(
        args.basis_order,
        quadrature_order=args.quadrature_order,
        relative_threshold=1e-14,
    )
    overlap, hamiltonian = agp_hamiltonian_transition_matrices(
        pairs, args.particles // 2, one_body, interaction
    )
    solved = solve_generalized_hermitian(hamiltonian, overlap)
    conditioning = {
        "balanced_overlap_condition_number": solved.retained_condition_number,
        "raw_overlap_condition_number": solved.raw_overlap_condition_number,
        "retained_rank": solved.retained_rank,
        "discarded_rank": solved.discarded_rank,
        "contribution_gram_spectrum": [
            float(x) for x in contribution_gram_spectrum(overlap, solved.amplitudes)
        ],
        "pruning_assessment": None,
        "pruning_or_restart_events": [],
    }
    point = soft_coulomb_point_from_training(
        args.point_id,
        training,
        conditioning,
        largest_basis_reference_energy=args.largest_basis_reference_energy,
        direct_dense_same_basis_reference_energy=(
            args.direct_dense_same_basis_reference_energy
        ),
        operator_error_estimate=args.operator_error_estimate,
    )
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_blind_single_agp",
        "training": training,
        "conditioning": conditioning,
        "normalized_point": point.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.point_id} error={training['error_vs_finite_basis']:.3e} "
        f"balanced_condition={conditioning['balanced_overlap_condition_number']:.3f}"
    )


if __name__ == "__main__":
    main()
