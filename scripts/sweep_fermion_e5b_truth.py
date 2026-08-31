"""Analytic and exterior-sector truth for six interacting harmonic fermions."""

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
    parser.add_argument("--basis-orders", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.05, 0.1, 0.35])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/fermion_e5b_truth_sweep.json"),
    )
    args = parser.parse_args()
    records = []
    for kappa in args.kappas:
        continuum = exact_interacting_harmonic_fermion_energy(6, kappa=kappa)
        basis_scan = []
        for dimension in args.basis_orders:
            if dimension < 6:
                raise ValueError("six fermions require D >= 6")
            started = time.perf_counter()
            one_body, interaction = harmonic_pair_hamiltonian(
                dimension, kappa=kappa, device="cpu"
            )
            hamiltonian = antisymmetric_many_body_hamiltonian(
                one_body, 6, interaction
            )
            energy = float(torch.linalg.eigvalsh(hamiltonian)[0].real)
            basis_scan.append(
                {
                    "D": dimension,
                    "exterior_dimension": math.comb(dimension, 6),
                    "finite_basis_energy": energy,
                    "error_vs_continuum": energy - continuum,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(
                f"kappa={kappa:g} D={dimension} E={energy:.12f} "
                f"error={energy-continuum:.3e}"
            )
        records.append(
            {
                "kappa": kappa,
                "continuum_energy": continuum,
                "basis_scan": basis_scan,
            }
        )
    result = {
        "schema_version": 1,
        "experiment": "fermion_e5b_six_particle_exterior_truth_sweep",
        "N": 6,
        "formula": "1/2 + 35/2*sqrt(1+6*kappa)",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
