"""Three-seed blind D=8, N=4, K=4 soft-Coulomb greedy sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from femps.algorithms import FiniteAgpConfig, canonical_pair_matrices, run_finite_agp_variable_projection


def _pairs(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return canonical_pair_matrices(payload["best_raw"])


def _config(k: int, frozen: int, steps: int, seed: int, device: str) -> FiniteAgpConfig:
    return FiniteAgpConfig(
        basis_order=8,
        particles=4,
        agp_terms=k,
        steps=steps,
        learning_rate=5e-3,
        final_learning_rate=1e-5,
        seed=seed,
        device=device,
        record_points=12,
        checkpoint_every=100,
        frozen_prefix_terms=frozen,
        interaction_model="soft_coulomb",
        soft_coulomb_quadrature_order=96,
        soft_coulomb_relative_threshold=1e-14,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[301, 302, 303])
    parser.add_argument("--k1-steps", type=int, default=400)
    parser.add_argument("--growth-steps", type=int, default=180)
    parser.add_argument("--joint-steps", type=int, default=180)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("docs/experiments/results/soft_coulomb_n4_seed_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/soft_coulomb_n4_k4_seed_sweep.json"),
    )
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_n4_d8_k4_three_seed_greedy_sweep",
        "runs": [],
    }
    for seed in args.seeds:
        seed_dir = args.checkpoint_directory / f"seed{seed}"
        k1_path = seed_dir / "k1_checkpoint.pt"
        k1 = run_finite_agp_variable_projection(
            _config(1, 0, args.k1_steps, seed, args.device),
            checkpoint_path=k1_path,
        )
        current = _pairs(k1_path)
        levels = []
        for target_k in (2, 3, 4):
            generator = torch.Generator().manual_seed(10000 * seed + target_k)
            random_pair = canonical_pair_matrices(
                torch.complex(
                    torch.randn(1, 8, 8, generator=generator, dtype=torch.float64),
                    torch.randn(1, 8, 8, generator=generator, dtype=torch.float64),
                )
            )
            initial = torch.cat((current, random_pair), dim=0)
            path = seed_dir / f"k{target_k}_growth_checkpoint.pt"
            growth = run_finite_agp_variable_projection(
                _config(
                    target_k,
                    target_k - 1,
                    args.growth_steps,
                    seed,
                    args.device,
                ),
                checkpoint_path=path,
                initial_pair_matrices=initial,
            )
            levels.append({"K": target_k, "growth": growth})
            current = _pairs(path)
        joint_path = seed_dir / "k4_joint_checkpoint.pt"
        joint = run_finite_agp_variable_projection(
            _config(4, 0, args.joint_steps, seed, args.device),
            checkpoint_path=joint_path,
            initial_pair_matrices=current,
        )
        result["runs"].append(
            {"seed": seed, "K1": k1, "levels": levels, "K4_joint": joint}
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"seed={seed} K1={k1['error_vs_finite_basis']:.3e} "
            f"K4={joint['error_vs_finite_basis']:.3e}"
        )


if __name__ == "__main__":
    main()
