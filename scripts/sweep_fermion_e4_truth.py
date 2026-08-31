"""Generate analytic and finite-basis exterior truth for the E4 benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian,
    exact_interacting_harmonic_fermion_energy,
    harmonic_pair_hamiltonian,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basis-orders", type=int, nargs="+", default=[4, 6, 8, 10, 12]
    )
    parser.add_argument(
        "--kappas", type=float, nargs="+", default=[0.0, 0.1, 0.35, 0.8]
    )
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/fermion_e4_truth_sweep.json"),
    )
    args = parser.parse_args()
    if any(order < 4 for order in args.basis_orders):
        raise ValueError("four fermions require every basis order to be at least four")

    records = []
    for kappa in args.kappas:
        continuum = exact_interacting_harmonic_fermion_energy(
            4, kappa=kappa, omega=args.omega
        )
        basis_records = []
        for dimension in args.basis_orders:
            started = time.perf_counter()
            one_body, interaction = harmonic_pair_hamiltonian(
                dimension,
                kappa=kappa,
                omega=args.omega,
                dtype=torch.complex128,
                device="cpu",
            )
            hamiltonian = antisymmetric_many_body_hamiltonian(
                one_body,
                particles=4,
                two_body=None if kappa == 0 else interaction,
            )
            energy = float(torch.linalg.eigvalsh(hamiltonian)[0].real)
            basis_records.append(
                {
                    "D": dimension,
                    "exterior_dimension": math.comb(dimension, 4),
                    "finite_basis_energy": energy,
                    "error_vs_continuum": abs(energy - continuum),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(
                f"kappa={kappa:g} D={dimension} E={energy:.12f} "
                f"error={abs(energy-continuum):.3e}"
            )
        records.append(
            {
                "kappa": kappa,
                "continuum_energy": continuum,
                "basis_scan": basis_records,
            }
        )
    result = {
        "schema_version": 1,
        "experiment": "fermion_e4_exterior_truth_sweep",
        "particles": 4,
        "omega": args.omega,
        "dtype": "complex128",
        "formula": "omega/2 + (N^2-1)/2*sqrt(omega^2+N*kappa)",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
