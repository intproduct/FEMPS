"""Scan E2 basis order and constrained Pfaffian pair rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from femps.algorithms import FactorizedPairConfig, run_factorized_pfaffian_pair
from femps.hamiltonians import (
    antisymmetric_two_particle_hamiltonian,
    exact_interacting_pair_energy,
    harmonic_pair_hamiltonian,
)
import torch


def ground_pair_matrix(one_body, interaction) -> torch.Tensor:
    hamiltonian = antisymmetric_two_particle_hamiltonian(one_body, interaction)
    _, vectors = torch.linalg.eigh(hamiltonian)
    coefficients = vectors[:, 0]
    dimension = one_body.shape[0]
    pair_matrix = torch.zeros(
        dimension, dimension, dtype=one_body.dtype, device=one_body.device
    )
    position = 0
    for first in range(dimension):
        for second in range(first + 1, dimension):
            pair_matrix[first, second] = coefficients[position]
            pair_matrix[second, first] = -coefficients[position]
            position += 1
    return pair_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=12)
    parser.add_argument("--pair-channels", default="1,2,3,4,5,6")
    parser.add_argument("--basis-values", default="4,6,8,10,12,14")
    parser.add_argument("--kappa", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/fermion_e2_pair_rank_sweep.json"),
    )
    args = parser.parse_args()
    channels = [int(value) for value in args.pair_channels.split(",")]
    basis_values = [int(value) for value in args.basis_values.split(",")]
    continuum = exact_interacting_pair_energy(kappa=args.kappa)
    basis_scan = []
    for order in basis_values:
        one_body, interaction = harmonic_pair_hamiltonian(order, kappa=args.kappa)
        exact = float(
            torch.linalg.eigvalsh(
                antisymmetric_two_particle_hamiltonian(one_body, interaction)
            )[0].real
        )
        basis_scan.append(
            {
                "D": order,
                "truncated_exact_energy": exact,
                "error_vs_continuum": abs(exact - continuum),
            }
        )
    rank_scan = []
    scan_one_body, scan_interaction = harmonic_pair_hamiltonian(
        args.basis_order, kappa=args.kappa
    )
    reference_pair = ground_pair_matrix(scan_one_body, scan_interaction)
    for channel_count in channels:
        print(
            f"run D={args.basis_order} pair_channels={channel_count}", flush=True
        )
        result = run_factorized_pfaffian_pair(
            FactorizedPairConfig(
                basis_order=args.basis_order,
                pair_channels=channel_count,
                kappa=args.kappa,
                steps=args.steps,
                seed=channel_count,
                device=args.device,
            ),
            initial_pair_matrix=reference_pair,
        )
        rank_scan.append(result)
        print(
            f"chi={channel_count} final={result['final_energy']:.12f} "
            f"error={result['error_vs_truncated']:.3e}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "experiment": "functional_pfaffian_e2_basis_pair_rank_sweep",
        "kappa": args.kappa,
        "continuum_reference_energy": continuum,
        "basis_scan": basis_scan,
        "pair_rank_scan": rank_scan,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
