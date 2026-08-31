"""Phase 8 interacting N=8 continuation from the noninteracting AGP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from femps.algorithms import (
    FiniteAgpConfig,
    canonical_pair_matrices,
    run_finite_agp_variable_projection,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=10)
    parser.add_argument("--kappa", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path("docs/experiments/results/fermion_e8a_checkpoint.pt"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("docs/experiments/results/fermion_e8b_checkpoint.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/fermion_e8b_single_agp.json"),
    )
    args = parser.parse_args()
    initial_pairs = None
    if not (args.resume and args.checkpoint.exists()):
        payload = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=False
        )
        initial_pairs = canonical_pair_matrices(payload["best_raw"])

    config = FiniteAgpConfig(
        basis_order=args.basis_order,
        particles=8,
        agp_terms=1,
        kappa=args.kappa,
        steps=args.steps,
        learning_rate=5e-3,
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
        initial_pair_matrices=initial_pairs,
    )
    result = {
        "schema_version": 1,
        "experiment": "fermion_e8b_interacting_single_agp",
        "initialization": "no_truth_continuation_from_e8a_noninteracting_state",
        "ordinary_particle_tensor_materialized": False,
        "training": training,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"E={training['final_energy']:.12f} "
        f"finite_error={training['error_vs_finite_basis']:.3e} "
        f"basis_error={training['basis_error_vs_continuum']:.3e}"
    )


if __name__ == "__main__":
    main()
