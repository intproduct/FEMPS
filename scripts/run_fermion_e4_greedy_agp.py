"""Greedily grow E4 from a blind K=1 state, then jointly relax K=2."""

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


def _random_pair(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(
        dimension, dimension, generator=generator, dtype=torch.float64
    )
    raw = torch.complex(real, imaginary).unsqueeze(0)
    return canonical_pair_matrices(raw)[0]


def _best_pairs(checkpoint: Path) -> torch.Tensor:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return canonical_pair_matrices(payload["best_raw"])


def _write(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=8)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--kappa", type=float, default=0.35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--growth-steps", type=int, default=300)
    parser.add_argument("--joint-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--k1-checkpoint",
        type=Path,
        default=Path(
            "docs/experiments/results/e4_k1_checkpoints/"
            "fermion_e4_variable_projection_seed0_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("docs/experiments/results/e4_greedy_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/fermion_e4_greedy_k2.json"),
    )
    args = parser.parse_args()
    first_pair = _best_pairs(args.k1_checkpoint)[0]
    result = {
        "schema_version": 1,
        "experiment": "fermion_e4_greedy_growth_then_joint_relaxation",
        "interpretation": (
            "The first AGP is obtained by blind K=1 energy optimization. The "
            "second is random and optimized without exact-state information."
        ),
        "runs": [],
    }
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    completed = {run["seed"] for run in result["runs"]}
    for seed in args.seeds:
        if seed in completed:
            print(f"skip completed seed={seed}")
            continue
        initial_pairs = torch.stack(
            (first_pair, _random_pair(args.basis_order, 20000 + seed))
        )
        growth_config = FiniteAgpConfig(
            basis_order=args.basis_order,
            particles=args.particles,
            agp_terms=2,
            kappa=args.kappa,
            steps=args.growth_steps,
            learning_rate=args.learning_rate,
            final_learning_rate=1e-5,
            seed=seed,
            device=args.device,
            record_points=20,
            checkpoint_every=100,
            frozen_prefix_terms=1,
        )
        growth_checkpoint = args.checkpoint_directory / (
            f"greedy_seed{seed}_checkpoint.pt"
        )
        growth = run_finite_agp_variable_projection(
            growth_config,
            checkpoint_path=growth_checkpoint,
            resume=args.resume and growth_checkpoint.exists(),
            initial_pair_matrices=(
                None
                if args.resume and growth_checkpoint.exists()
                else initial_pairs
            ),
        )
        grown_pairs = _best_pairs(growth_checkpoint)
        joint_config = FiniteAgpConfig(
            basis_order=args.basis_order,
            particles=args.particles,
            agp_terms=2,
            kappa=args.kappa,
            steps=args.joint_steps,
            learning_rate=args.learning_rate,
            final_learning_rate=1e-5,
            seed=seed,
            device=args.device,
            record_points=20,
            checkpoint_every=100,
            frozen_prefix_terms=0,
        )
        joint_checkpoint = args.checkpoint_directory / (
            f"joint_seed{seed}_checkpoint.pt"
        )
        joint = run_finite_agp_variable_projection(
            joint_config,
            checkpoint_path=joint_checkpoint,
            resume=args.resume and joint_checkpoint.exists(),
            initial_pair_matrices=(
                None if args.resume and joint_checkpoint.exists() else grown_pairs
            ),
        )
        result["runs"].append(
            {"seed": seed, "growth": growth, "joint": joint}
        )
        _write(args.output, result)
        print(
            f"seed={seed} growth_error={growth['error_vs_finite_basis']:.3e} "
            f"joint_error={joint['error_vs_finite_basis']:.3e}"
        )


if __name__ == "__main__":
    main()
