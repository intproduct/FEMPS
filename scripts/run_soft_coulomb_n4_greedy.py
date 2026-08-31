"""Greedy K=1 to K=2 soft-Coulomb finite-AGP growth."""

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


def _checkpoint_pairs(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload["best_raw"]
    return canonical_pair_matrices(raw.unsqueeze(0) if raw.ndim == 2 else raw)


def _config(*, steps: int, frozen: int, seed: int, device: str) -> FiniteAgpConfig:
    return FiniteAgpConfig(
        basis_order=8,
        particles=4,
        agp_terms=2,
        steps=steps,
        learning_rate=5e-3,
        final_learning_rate=1e-5,
        seed=seed,
        device=device,
        record_points=20,
        checkpoint_every=100,
        frozen_prefix_terms=frozen,
        interaction_model="soft_coulomb",
        soft_coulomb_quadrature_order=96,
        soft_coulomb_relative_threshold=1e-14,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--k1-checkpoint",
        type=Path,
        default=Path("docs/experiments/results/soft_coulomb_n4_checkpoint.pt"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("docs/experiments/results/soft_coulomb_n4_greedy_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/soft_coulomb_n4_greedy_k2.json"),
    )
    args = parser.parse_args()
    first = _checkpoint_pairs(args.k1_checkpoint)[0]
    generator = torch.Generator().manual_seed(args.seed)
    random_raw = torch.complex(
        torch.randn(8, 8, generator=generator, dtype=torch.float64),
        torch.randn(8, 8, generator=generator, dtype=torch.float64),
    ).unsqueeze(0)
    second = canonical_pair_matrices(random_raw)[0]
    initial = torch.stack((first, second))
    growth_checkpoint = args.checkpoint_directory / "growth_checkpoint.pt"
    growth = run_finite_agp_variable_projection(
        _config(steps=args.steps, frozen=1, seed=args.seed, device=args.device),
        checkpoint_path=growth_checkpoint,
        initial_pair_matrices=initial,
    )
    grown = _checkpoint_pairs(growth_checkpoint)
    joint_checkpoint = args.checkpoint_directory / "joint_checkpoint.pt"
    joint = run_finite_agp_variable_projection(
        _config(steps=args.steps, frozen=0, seed=args.seed, device=args.device),
        checkpoint_path=joint_checkpoint,
        initial_pair_matrices=grown,
    )
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_n4_greedy_k2",
        "growth": growth,
        "joint": joint,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"growth_error={growth['error_vs_finite_basis']:.3e} "
        f"joint_error={joint['error_vs_finite_basis']:.3e}"
    )


if __name__ == "__main__":
    main()
