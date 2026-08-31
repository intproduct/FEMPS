"""Embed conditioned D=10 K=5 chains in D=12 and refine them."""

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
    if source.shape[1] == dimension:
        return source
    if source.shape[1] > dimension:
        raise ValueError("source basis is larger than the target basis")
    embedded = torch.zeros(
        source.shape[0], dimension, dimension, dtype=source.dtype
    )
    embedded[:, : source.shape[1], : source.shape[2]] = source
    return embedded


def _conditioning(pairs: torch.Tensor) -> dict:
    one_body = harmonic_hamiltonian(12, dtype=torch.complex128)
    interaction, _ = soft_coulomb_operator(
        12, quadrature_order=128, relative_threshold=1e-14
    )
    overlap, hamiltonian = agp_hamiltonian_transition_matrices(
        pairs, 2, one_body, interaction
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[301, 302, 303])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_conditioned_d10_k5_checkpoints"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_conditioned_d12_k5_checkpoints"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_conditioned_d12_k5_seeds.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        result = {
            "schema_version": 1,
            "experiment": "soft_coulomb_conditioned_d12_k5_three_seed_refinement",
            "runs": [],
        }
    completed_seeds = {run["seed"] for run in result["runs"]}
    for seed in args.seeds:
        if seed in completed_seeds:
            print(f"seed={seed} already complete; skipping")
            continue
        initial = _pairs(args.source_directory / f"seed{seed}_joint.pt", 12)
        config = FiniteAgpConfig(
            basis_order=12,
            particles=4,
            agp_terms=5,
            steps=args.steps,
            learning_rate=1e-3,
            final_learning_rate=1e-5,
            seed=seed,
            device=args.device,
            record_points=20,
            checkpoint_every=100,
            interaction_model="soft_coulomb",
            soft_coulomb_quadrature_order=128,
            soft_coulomb_relative_threshold=1e-14,
        )
        checkpoint = args.checkpoint_directory / f"seed{seed}_checkpoint.pt"
        training = run_finite_agp_variable_projection(
            config,
            checkpoint_path=checkpoint,
            initial_pair_matrices=initial,
        )
        conditioning = _conditioning(_pairs(checkpoint, 12))
        point = soft_coulomb_point_from_training(
            f"n4-d12-k5-s{seed}",
            training,
            conditioning,
            largest_basis_reference_energy=11.023082853675,
            direct_dense_same_basis_reference_energy=11.023094656411,
            operator_error_estimate=6e-14,
        )
        result["runs"].append(
            {
                "seed": seed,
                "training": training,
                "conditioning": conditioning,
                "normalized_point": point.to_dict(),
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"seed={seed} error={training['error_vs_finite_basis']:.3e} "
            f"balanced_condition={conditioning['balanced_overlap_condition_number']:.3f}"
        )


if __name__ == "__main__":
    main()
