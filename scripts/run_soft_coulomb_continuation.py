"""Refine or grow one finite-AGP soft-Coulomb checkpoint at a new basis order."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from femps.algorithms import (
    FiniteAgpConfig,
    assess_term_pruning,
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


def _pairs(path: Path, dimension: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = canonical_pair_matrices(payload["best_raw"])
    if source.shape[1] > dimension:
        raise ValueError("source basis exceeds target basis")
    if source.shape[1] == dimension:
        return source
    embedded = torch.zeros(
        source.shape[0], dimension, dimension, dtype=source.dtype
    )
    embedded[:, : source.shape[1], : source.shape[2]] = source
    return embedded


def _config(
    args: argparse.Namespace, *, terms: int, steps: int, frozen: int, learning_rate: float
) -> FiniteAgpConfig:
    return FiniteAgpConfig(
        basis_order=args.basis_order,
        particles=args.particles,
        agp_terms=terms,
        steps=steps,
        learning_rate=learning_rate,
        final_learning_rate=args.final_learning_rate,
        seed=args.seed,
        device=args.device,
        record_points=20,
        checkpoint_every=100,
        frozen_prefix_terms=frozen,
        interaction_model="soft_coulomb",
        soft_coulomb_quadrature_order=args.quadrature_order,
        soft_coulomb_relative_threshold=1e-14,
    )


def _conditioning(args: argparse.Namespace, pairs: torch.Tensor) -> dict:
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
    return {
        "balanced_overlap_condition_number": solved.retained_condition_number,
        "raw_overlap_condition_number": solved.raw_overlap_condition_number,
        "retained_rank": solved.retained_rank,
        "discarded_rank": solved.discarded_rank,
        "contribution_gram_spectrum": [
            float(x) for x in contribution_gram_spectrum(overlap, solved.amplitudes)
        ],
        "pruning_assessment": asdict(
            assess_term_pruning(hamiltonian, overlap)
        ),
        "pruning_or_restart_events": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-id", required=True)
    parser.add_argument("--particles", type=int, required=True)
    parser.add_argument("--basis-order", type=int, required=True)
    parser.add_argument("--target-terms", type=int, required=True)
    parser.add_argument("--quadrature-order", type=int, default=128)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--growth-steps", type=int, default=180)
    parser.add_argument("--joint-steps", type=int, default=200)
    parser.add_argument("--growth-learning-rate", type=float, default=2e-3)
    parser.add_argument("--joint-learning-rate", type=float, default=1e-3)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--largest-basis-reference-energy", type=float)
    parser.add_argument("--direct-dense-same-basis-reference-energy", type=float)
    parser.add_argument("--operator-error-estimate", type=float)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    current = _pairs(args.source_checkpoint, args.basis_order)
    source_terms = current.shape[0]
    if args.target_terms not in (source_terms, source_terms + 1):
        raise ValueError("target_terms must equal source K or source K + 1")
    stages = {}
    if args.target_terms == source_terms + 1:
        generator = torch.Generator().manual_seed(100000 + args.seed + args.target_terms)
        new_pair = canonical_pair_matrices(
            torch.complex(
                torch.randn(
                    1,
                    args.basis_order,
                    args.basis_order,
                    generator=generator,
                    dtype=torch.float64,
                ),
                torch.randn(
                    1,
                    args.basis_order,
                    args.basis_order,
                    generator=generator,
                    dtype=torch.float64,
                ),
            )
        )
        current = torch.cat((current, new_pair), dim=0)
        growth_path = args.checkpoint_directory / "growth_checkpoint.pt"
        stages["growth"] = run_finite_agp_variable_projection(
            _config(
                args,
                terms=args.target_terms,
                steps=args.growth_steps,
                frozen=source_terms,
                learning_rate=args.growth_learning_rate,
            ),
            checkpoint_path=growth_path,
            initial_pair_matrices=current,
        )
        current = _pairs(growth_path, args.basis_order)

    joint_path = args.checkpoint_directory / "joint_checkpoint.pt"
    stages["joint"] = run_finite_agp_variable_projection(
        _config(
            args,
            terms=args.target_terms,
            steps=args.joint_steps,
            frozen=0,
            learning_rate=args.joint_learning_rate,
        ),
        checkpoint_path=joint_path,
        initial_pair_matrices=current,
    )
    conditioning = _conditioning(args, _pairs(joint_path, args.basis_order))
    point = soft_coulomb_point_from_training(
        args.point_id,
        stages["joint"],
        conditioning,
        largest_basis_reference_energy=args.largest_basis_reference_energy,
        direct_dense_same_basis_reference_energy=(
            args.direct_dense_same_basis_reference_energy
        ),
        operator_error_estimate=args.operator_error_estimate,
    )
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_finite_agp_checkpoint_continuation",
        "source_checkpoint": str(args.source_checkpoint),
        "stages": stages,
        "conditioning": conditioning,
        "normalized_point": point.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.point_id} error={stages['joint']['error_vs_finite_basis']:.3e} "
        f"balanced_condition={conditioning['balanced_overlap_condition_number']:.3f}"
    )


if __name__ == "__main__":
    main()
