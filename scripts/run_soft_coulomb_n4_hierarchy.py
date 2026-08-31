"""Grow the D=8 soft-Coulomb N=4 finite-AGP hierarchy from K=2 to K=4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from femps.algorithms import FiniteAgpConfig, canonical_pair_matrices, run_finite_agp_variable_projection


def _pairs(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return canonical_pair_matrices(payload["best_raw"])


def _config(dimension: int, quadrature: int, k: int, frozen: int, steps: int, seed: int, device: str) -> FiniteAgpConfig:
    return FiniteAgpConfig(
        basis_order=dimension,
        particles=4,
        agp_terms=k,
        steps=steps,
        learning_rate=4e-3,
        final_learning_rate=1e-5,
        seed=seed,
        device=device,
        record_points=20,
        checkpoint_every=100,
        frozen_prefix_terms=frozen,
        interaction_model="soft_coulomb",
        soft_coulomb_quadrature_order=quadrature,
        soft_coulomb_relative_threshold=1e-14,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=8)
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--k2-checkpoint",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_n4_greedy_checkpoints/"
            "joint_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("docs/experiments/results/soft_coulomb_n4_hierarchy_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/soft_coulomb_n4_k_hierarchy.json"),
    )
    args = parser.parse_args()
    current = _pairs(args.k2_checkpoint)
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_n4_d8_greedy_k_hierarchy",
        "starting_K": 2,
        "levels": [],
    }
    for target_k in (3, 4):
        generator = torch.Generator().manual_seed(args.seed + target_k)
        real = torch.randn(1, args.basis_order, args.basis_order, generator=generator, dtype=torch.float64)
        imaginary = torch.randn(1, args.basis_order, args.basis_order, generator=generator, dtype=torch.float64)
        new_pair = canonical_pair_matrices(torch.complex(real, imaginary))
        initial = torch.cat((current, new_pair), dim=0)
        growth_path = args.checkpoint_directory / f"k{target_k}_growth_checkpoint.pt"
        growth = run_finite_agp_variable_projection(
            _config(args.basis_order, args.quadrature_order, target_k, target_k - 1, args.steps, args.seed, args.device),
            checkpoint_path=growth_path,
            initial_pair_matrices=initial,
        )
        grown = _pairs(growth_path)
        joint_path = args.checkpoint_directory / f"k{target_k}_joint_checkpoint.pt"
        joint = run_finite_agp_variable_projection(
            _config(args.basis_order, args.quadrature_order, target_k, 0, args.steps, args.seed, args.device),
            checkpoint_path=joint_path,
            initial_pair_matrices=grown,
        )
        result["levels"].append(
            {"K": target_k, "growth": growth, "joint": joint}
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        current = _pairs(joint_path)
        print(
            f"K={target_k} growth_error={growth['error_vs_finite_basis']:.3e} "
            f"joint_error={joint['error_vs_finite_basis']:.3e}"
        )


if __name__ == "__main__":
    main()
