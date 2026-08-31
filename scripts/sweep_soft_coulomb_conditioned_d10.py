"""Refine three independent D=8 K=4 chains in the conditioned D=10 solver."""

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
from femps.hamiltonians import (
    agp_hamiltonian_transition_matrices,
    soft_coulomb_operator,
)


def _embedded_pairs(path: Path, dimension: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    pairs = canonical_pair_matrices(payload["best_raw"])
    embedded = torch.zeros(
        pairs.shape[0], dimension, dimension, dtype=pairs.dtype
    )
    embedded[:, : pairs.shape[1], : pairs.shape[2]] = pairs
    return embedded


def _diagnostics(pairs: torch.Tensor, quadrature: int) -> dict:
    dimension = pairs.shape[1]
    one_body = harmonic_hamiltonian(dimension, dtype=torch.complex128)
    interaction, _ = soft_coulomb_operator(
        dimension, quadrature_order=quadrature, relative_threshold=1e-14
    )
    overlap, hamiltonian = agp_hamiltonian_transition_matrices(
        pairs, 2, one_body, interaction
    )
    solved = solve_generalized_hermitian(hamiltonian, overlap)
    pruning = assess_term_pruning(hamiltonian, overlap)
    return {
        "balanced_overlap_condition_number": solved.retained_condition_number,
        "raw_overlap_condition_number": solved.raw_overlap_condition_number,
        "retained_rank": solved.retained_rank,
        "discarded_rank": solved.discarded_rank,
        "contribution_gram_spectrum": [
            float(x) for x in contribution_gram_spectrum(overlap, solved.amplitudes)
        ],
        "pruning_assessment": asdict(pruning),
        "pruning_or_restart_events": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[301, 302, 303])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_n4_seed_checkpoints"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_conditioned_d10_checkpoints"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_conditioned_d10_k4_seeds.json"
        ),
    )
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_conditioned_d10_k4_three_seed_refinement",
        "initialization": "independent_blind_D8_chains_embedded_in_D10",
        "runs": [],
    }
    for seed in args.seeds:
        source = args.source_directory / f"seed{seed}" / "k4_joint_checkpoint.pt"
        initial = _embedded_pairs(source, 10)
        config = FiniteAgpConfig(
            basis_order=10,
            particles=4,
            agp_terms=4,
            steps=args.steps,
            learning_rate=2e-3,
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
        run = run_finite_agp_variable_projection(
            config,
            checkpoint_path=checkpoint,
            initial_pair_matrices=initial,
        )
        final_pairs = _embedded_pairs(checkpoint, 10)
        diagnostics = _diagnostics(final_pairs, 128)
        result["runs"].append(
            {"seed": seed, "training": run, "conditioning": diagnostics}
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"seed={seed} error={run['error_vs_finite_basis']:.3e} "
            f"balanced_condition="
            f"{diagnostics['balanced_overlap_condition_number']:.3f}"
        )


if __name__ == "__main__":
    main()
