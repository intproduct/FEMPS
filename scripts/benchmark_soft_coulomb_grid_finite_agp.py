"""Matched-grid finite-AGP, ordered-sector, exterior-CI, and TT comparison."""

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
from femps.hamiltonians import (
    FactorizedTwoBodyOperator,
    antisymmetric_many_body_hamiltonian,
)
from femps.ordered_sector import (
    finite_difference_harmonic_hamiltonian,
    ordered_sector_hamiltonian,
)


def _grid_operators(
    grid_points: int, spacing: float
) -> tuple[torch.Tensor, FactorizedTwoBodyOperator, torch.Tensor]:
    grid, one_body = finite_difference_harmonic_hamiltonian(
        grid_points, spacing, dtype=torch.float64
    )
    potential = 1 / torch.sqrt((grid[:, None] - grid[None, :]) ** 2 + 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(potential)
    factors = torch.diag_embed(eigenvectors.transpose(0, 1))
    interaction = FactorizedTwoBodyOperator(factors, factors, eigenvalues)
    return one_body, interaction, potential


def _pairs(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return canonical_pair_matrices(payload["best_raw"])


def _config(
    *,
    dimension: int,
    particles: int,
    terms: int,
    frozen: int,
    steps: int,
    seed: int,
    device: str,
) -> FiniteAgpConfig:
    return FiniteAgpConfig(
        basis_order=dimension,
        particles=particles,
        agp_terms=terms,
        steps=steps,
        learning_rate=2e-3 if frozen else 1e-3,
        final_learning_rate=1e-5,
        seed=seed,
        device=device,
        record_points=15,
        checkpoint_every=100,
        frozen_prefix_terms=frozen,
        interaction_model="soft_coulomb",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--grid-points", type=int, default=8)
    parser.add_argument("--spacing", type=float, default=0.7)
    parser.add_argument("--max-terms", type=int, default=3)
    parser.add_argument("--k1-steps", type=int, default=250)
    parser.add_argument("--growth-steps", type=int, default=150)
    parser.add_argument("--joint-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_grid_finite_agp_checkpoints"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_grid_finite_agp.json"
        ),
    )
    args = parser.parse_args()
    one_body, interaction, potential = _grid_operators(
        args.grid_points, args.spacing
    )
    operator_id = (
        f"finite-difference-soft-coulomb-n{args.particles}-"
        f"d{args.grid_points}-dx{args.spacing:g}-v1"
    )
    ordered = ordered_sector_hamiltonian(
        one_body, args.particles, pair_potential=potential
    )
    exterior = antisymmetric_many_body_hamiltonian(
        one_body, args.particles, interaction
    )
    truth_energy = float(torch.linalg.eigvalsh(ordered)[0])
    common = {
        "operators": (one_body, interaction),
        "operator_id": operator_id,
    }
    k1_path = args.checkpoint_directory / "k1_checkpoint.pt"
    k1 = run_finite_agp_variable_projection(
        _config(
            dimension=args.grid_points,
            particles=args.particles,
            terms=1,
            frozen=0,
            steps=args.k1_steps,
            seed=args.seed,
            device=args.device,
        ),
        checkpoint_path=k1_path,
        **common,
    )
    current = _pairs(k1_path)
    levels = [{"K": 1, "joint": k1}]
    for terms in range(2, args.max_terms + 1):
        generator = torch.Generator().manual_seed(1000 * args.seed + terms)
        new_pair = canonical_pair_matrices(
            torch.complex(
                torch.randn(
                    1,
                    args.grid_points,
                    args.grid_points,
                    generator=generator,
                    dtype=torch.float64,
                ),
                torch.randn(
                    1,
                    args.grid_points,
                    args.grid_points,
                    generator=generator,
                    dtype=torch.float64,
                ),
            )
        )
        initial = torch.cat((current, new_pair), dim=0)
        growth_path = args.checkpoint_directory / f"k{terms}_growth.pt"
        growth = run_finite_agp_variable_projection(
            _config(
                dimension=args.grid_points,
                particles=args.particles,
                terms=terms,
                frozen=terms - 1,
                steps=args.growth_steps,
                seed=args.seed,
                device=args.device,
            ),
            checkpoint_path=growth_path,
            initial_pair_matrices=initial,
            **common,
        )
        joint_path = args.checkpoint_directory / f"k{terms}_joint.pt"
        joint = run_finite_agp_variable_projection(
            _config(
                dimension=args.grid_points,
                particles=args.particles,
                terms=terms,
                frozen=0,
                steps=args.joint_steps,
                seed=args.seed,
                device=args.device,
            ),
            checkpoint_path=joint_path,
            initial_pair_matrices=_pairs(growth_path),
            **common,
        )
        levels.append({"K": terms, "growth": growth, "joint": joint})
        current = _pairs(joint_path)
        print(f"K={terms} error={joint['error_vs_finite_basis']:.3e}")
    result = {
        "schema_version": 1,
        "experiment": "matched_grid_finite_agp_ordered_exterior_tt",
        "operator_id": operator_id,
        "N": args.particles,
        "D": args.grid_points,
        "spacing": args.spacing,
        "ordered_exterior_matrix_max_absolute_difference": float(
            torch.max(torch.abs(ordered - exterior))
        ),
        "direct_truth_energy": truth_energy,
        "levels": levels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
