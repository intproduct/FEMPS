"""Check soft-Coulomb quadrature convergence at production basis orders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from femps.basis import harmonic_hamiltonian
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian_dense_two_body,
    soft_coulomb_dense_quadrature,
)


def _point(dimension: int, quadrature: int) -> dict:
    started = time.perf_counter()
    one_body = harmonic_hamiltonian(dimension, dtype=torch.complex128)
    interaction = soft_coulomb_dense_quadrature(
        dimension, quadrature_order=quadrature
    )
    hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, 4, interaction
    )
    return {
        "D": dimension,
        "Q": quadrature,
        "exterior_dimension": hamiltonian.shape[0],
        "ground_energy": float(torch.linalg.eigvalsh(hamiltonian)[0].real),
        "hermiticity_residual": float(
            torch.linalg.vector_norm(hamiltonian - hamiltonian.mH)
            / torch.linalg.vector_norm(hamiltonian)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-orders", nargs="+", type=int, default=[8, 10, 12])
    parser.add_argument("--quadrature-orders", nargs="+", type=int, default=[96, 128, 160])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_quadrature_by_basis.json"
        ),
    )
    args = parser.parse_args()
    scans = []
    for dimension in args.basis_orders:
        points = [_point(dimension, order) for order in args.quadrature_orders]
        reference = points[-1]["ground_energy"]
        for point in points:
            point["energy_difference_vs_largest_Q"] = (
                point["ground_energy"] - reference
            )
        scans.append({"D": dimension, "points": points})
        print(
            f"D={dimension} Q={args.quadrature_orders[-2]}-"
            f"Q={args.quadrature_orders[-1]}: "
            f"{points[-2]['energy_difference_vs_largest_Q']:.3e}"
        )
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_quadrature_convergence_by_basis",
        "hamiltonian": "spin-polarized N=4, g=a=omega=1 on R",
        "truth_path": "direct four-index Slater-Condon exterior Hamiltonian",
        "largest_Q_is_a_numerical_reference_not_an_exact_operator": True,
        "scans": scans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
