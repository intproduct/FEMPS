"""Oracle-initialized finite-AGP expressivity hierarchy for E4.

The exact finite-basis ground vector is used only to fit the AGP span.  This is
a representation diagnostic and initializer, not a blind variational solve.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import time

import torch

from femps.exterior import agp_sum_norm, agp_tensor, particle_tt_ranks
from femps.hamiltonians import (
    agp_sum_energy,
    antisymmetric_many_body_hamiltonian,
    exact_interacting_harmonic_fermion_energy,
    harmonic_pair_hamiltonian,
)


def _normalized_pairs(raw: torch.Tensor) -> torch.Tensor:
    skew = raw - raw.transpose(1, 2)
    return skew / torch.linalg.vector_norm(skew, dim=(1, 2))[:, None, None]


def _four_form_basis(
    raw: torch.Tensor, supports: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_matrices = _normalized_pairs(raw)
    first, second, third, fourth = supports.transpose(0, 1)
    coefficients = (
        pair_matrices[:, first, second] * pair_matrices[:, third, fourth]
        - pair_matrices[:, first, third] * pair_matrices[:, second, fourth]
        + pair_matrices[:, first, fourth] * pair_matrices[:, second, third]
    )
    return pair_matrices, coefficients


def _optimal_span_state(
    raw: torch.Tensor,
    supports: torch.Tensor,
    target: torch.Tensor,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pair_matrices, basis = _four_form_basis(raw, supports)
    gram = basis @ basis.transpose(0, 1)
    projection = basis @ target
    amplitudes = torch.linalg.solve(
        gram
        + regularization
        * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device),
        projection,
    )
    state = amplitudes @ basis
    fidelity = (state @ target).square() / ((state @ state) * (target @ target))
    return 1 - fidelity, pair_matrices, amplitudes, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=8)
    parser.add_argument("--kappa", type=float, default=0.35)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--final-learning-rate", type=float, default=1e-4)
    parser.add_argument("--regularization", type=float, default=1e-12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/fermion_e4_agp_rank_sweep.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "docs/experiments/results/fermion_e4_agp_rank_checkpoint.pt"
        ),
    )
    args = parser.parse_args()
    if args.basis_order < 4 or any(length < 1 for length in args.lengths):
        raise ValueError("require D >= 4 and positive AGP lengths")
    lengths = sorted(set(args.lengths))
    dimension = args.basis_order
    one_body, interaction = harmonic_pair_hamiltonian(
        dimension, kappa=args.kappa, device="cpu"
    )
    hamiltonian = antisymmetric_many_body_hamiltonian(
        one_body, 4, interaction
    ).real
    eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
    finite_energy = float(eigenvalues[0])
    target = eigenvectors[:, 0]
    supports = torch.tensor(
        list(itertools.combinations(range(dimension), 4)), dtype=torch.long
    )
    records = []
    best_by_length: dict[int, dict[str, torch.Tensor | float | int]] = {}
    for seed in args.seeds:
        previous_raw = None
        for length in lengths:
            generator = torch.Generator().manual_seed(1000 * seed + length)
            initial = torch.randn(
                length,
                dimension,
                dimension,
                generator=generator,
                dtype=torch.float64,
            )
            if previous_raw is not None:
                retained = min(previous_raw.shape[0], length)
                initial[:retained] = previous_raw[:retained]
            raw = torch.nn.Parameter(initial)
            optimizer = torch.optim.Adam([raw], lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.steps,
                eta_min=args.final_learning_rate,
            )
            started = time.perf_counter()
            for _ in range(args.steps):
                optimizer.zero_grad()
                loss, _, _, _ = _optimal_span_state(
                    raw, supports, target, args.regularization
                )
                loss.backward()
                optimizer.step()
                scheduler.step()
            elapsed = time.perf_counter() - started
            with torch.no_grad():
                loss, pair_matrices, amplitudes, state = _optimal_span_state(
                    raw, supports, target, args.regularization
                )
                norm = state @ state
                energy = float((state @ (hamiltonian @ state)) / norm)
            record = {
                "K": length,
                "seed": seed,
                "infidelity": float(loss),
                "explicit_exterior_energy": energy,
                "error_vs_finite_basis": energy - finite_energy,
                "elapsed_seconds": elapsed,
            }
            records.append(record)
            current_best = best_by_length.get(length)
            if current_best is None or float(loss) < float(current_best["infidelity"]):
                best_by_length[length] = {
                    "seed": seed,
                    "infidelity": float(loss),
                    "pair_matrices": pair_matrices.detach().clone(),
                    "amplitudes": amplitudes.detach().clone(),
                    "state": state.detach().clone(),
                    "energy": energy,
                }
            previous_raw = raw.detach().clone()
            print(
                f"seed={seed} K={length} infidelity={float(loss):.3e} "
                f"finite_error={energy-finite_energy:.3e}"
            )

    best_records = []
    checkpoint = {}
    for length in lengths:
        best = best_by_length[length]
        pair_matrices_real = best["pair_matrices"]
        amplitudes_real = best["amplitudes"]
        assert isinstance(pair_matrices_real, torch.Tensor)
        assert isinstance(amplitudes_real, torch.Tensor)
        pair_matrices = torch.complex(
            pair_matrices_real, torch.zeros_like(pair_matrices_real)
        )
        amplitudes = torch.complex(
            amplitudes_real, torch.zeros_like(amplitudes_real)
        )
        started = time.perf_counter()
        polynomial_energy = float(
            agp_sum_energy(
                pair_matrices, amplitudes, 2, one_body, interaction
            ).detach()
        )
        polynomial_seconds = time.perf_counter() - started
        polynomial_norm = float(
            agp_sum_norm(pair_matrices, amplitudes, 2).detach()
        )
        particle_state = sum(
            amplitudes[term] * agp_tensor(pair_matrices[term], 2)
            for term in range(length)
        )
        best_records.append(
            {
                "K": length,
                "selected_seed": int(best["seed"]),
                "infidelity": float(best["infidelity"]),
                "explicit_exterior_energy": float(best["energy"]),
                "polynomial_energy": polynomial_energy,
                "polynomial_explicit_absolute_difference": abs(
                    polynomial_energy - float(best["energy"])
                ),
                "error_vs_finite_basis": polynomial_energy - finite_energy,
                "error_vs_continuum": polynomial_energy
                - exact_interacting_harmonic_fermion_energy(
                    4, kappa=args.kappa
                ),
                "polynomial_norm": polynomial_norm,
                "polynomial_evaluation_seconds": polynomial_seconds,
                "ordinary_particle_tt_ranks": list(
                    particle_tt_ranks(particle_state)
                ),
            }
        )
        checkpoint[length] = {
            "pair_matrices": pair_matrices_real,
            "amplitudes": amplitudes_real,
            "selected_seed": int(best["seed"]),
        }
    result = {
        "schema_version": 1,
        "experiment": "fermion_e4_oracle_finite_agp_rank_sweep",
        "interpretation": (
            "Exact finite-basis eigenvectors are used for span fitting; this is "
            "a representation diagnostic and initializer, not blind optimization."
        ),
        "D": dimension,
        "N": 4,
        "kappa": args.kappa,
        "continuum_reference_energy": exact_interacting_harmonic_fermion_energy(
            4, kappa=args.kappa
        ),
        "finite_basis_reference_energy": finite_energy,
        "exterior_dimension": math.comb(dimension, 4),
        "steps": args.steps,
        "seeds": args.seeds,
        "all_runs": records,
        "best_by_K": best_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {"schema_version": 1, "config": vars(args), "states": checkpoint},
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
