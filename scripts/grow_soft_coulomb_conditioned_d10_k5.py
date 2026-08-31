"""Test reproducible K=4 to K=5 growth for three conditioned D=10 chains."""

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


def _pairs(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return canonical_pair_matrices(payload["best_raw"])


def _config(*, seed: int, steps: int, frozen: int, device: str) -> FiniteAgpConfig:
    return FiniteAgpConfig(
        basis_order=10,
        particles=4,
        agp_terms=5,
        steps=steps,
        learning_rate=2e-3 if frozen else 1e-3,
        final_learning_rate=1e-5,
        seed=seed,
        device=device,
        record_points=15,
        checkpoint_every=100,
        frozen_prefix_terms=frozen,
        interaction_model="soft_coulomb",
        soft_coulomb_quadrature_order=128,
        soft_coulomb_relative_threshold=1e-14,
    )


def _conditioning(pairs: torch.Tensor) -> dict:
    one_body = harmonic_hamiltonian(10, dtype=torch.complex128)
    interaction, _ = soft_coulomb_operator(
        10, quadrature_order=128, relative_threshold=1e-14
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
    parser.add_argument("--growth-steps", type=int, default=150)
    parser.add_argument("--joint-steps", type=int, default=150)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--k4-checkpoint-directory",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_conditioned_d10_checkpoints"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_conditioned_d10_k5_checkpoints"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_conditioned_d10_k5_seeds.json"
        ),
    )
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_conditioned_d10_k5_three_seed_growth",
        "runs": [],
    }
    for seed in args.seeds:
        current = _pairs(
            args.k4_checkpoint_directory / f"seed{seed}_checkpoint.pt"
        )
        generator = torch.Generator().manual_seed(50000 + seed)
        new_pair = canonical_pair_matrices(
            torch.complex(
                torch.randn(1, 10, 10, generator=generator, dtype=torch.float64),
                torch.randn(1, 10, 10, generator=generator, dtype=torch.float64),
            )
        )
        initial = torch.cat((current, new_pair), dim=0)
        growth_path = args.checkpoint_directory / f"seed{seed}_growth.pt"
        growth = run_finite_agp_variable_projection(
            _config(
                seed=seed,
                steps=args.growth_steps,
                frozen=4,
                device=args.device,
            ),
            checkpoint_path=growth_path,
            initial_pair_matrices=initial,
        )
        joint_path = args.checkpoint_directory / f"seed{seed}_joint.pt"
        joint = run_finite_agp_variable_projection(
            _config(
                seed=seed,
                steps=args.joint_steps,
                frozen=0,
                device=args.device,
            ),
            checkpoint_path=joint_path,
            initial_pair_matrices=_pairs(growth_path),
        )
        conditioning = _conditioning(_pairs(joint_path))
        result["runs"].append(
            {
                "seed": seed,
                "growth": growth,
                "joint": joint,
                "conditioning": conditioning,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"seed={seed} initial_K5_error="
            f"{growth['initial_energy'] - growth['finite_basis_reference_energy']:.3e} "
            f"K5_error={joint['error_vs_finite_basis']:.3e} "
            f"balanced_condition={conditioning['balanced_overlap_condition_number']:.3f}"
        )


if __name__ == "__main__":
    main()
