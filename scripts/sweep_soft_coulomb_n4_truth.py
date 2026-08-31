"""Independent N=4 soft-Coulomb basis and quadrature truth sweep."""

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


def _point(dimension: int, quadrature_order: int) -> dict:
    started = time.perf_counter()
    one_body = harmonic_hamiltonian(dimension, dtype=torch.complex128)
    interaction = soft_coulomb_dense_quadrature(
        dimension,
        quadrature_order=quadrature_order,
        coupling=1.0,
        softening=1.0,
    )
    hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, 4, interaction
    )
    hermiticity = float(
        torch.linalg.vector_norm(hamiltonian - hamiltonian.mH)
        / torch.linalg.vector_norm(hamiltonian)
    )
    energy = float(torch.linalg.eigvalsh(hamiltonian)[0].real)
    return {
        "D": dimension,
        "Q": quadrature_order,
        "exterior_dimension": hamiltonian.shape[0],
        "ground_energy": energy,
        "hamiltonian_hermiticity_residual": hermiticity,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/soft_coulomb_n4_truth_sweep.json"),
    )
    args = parser.parse_args()
    quadrature_scan = [_point(8, order) for order in (48, 64, 96, 128)]
    basis_scan = [_point(order, 128) for order in (4, 6, 8, 10, 12, 14)]
    q_reference = quadrature_scan[-1]["ground_energy"]
    for point in quadrature_scan:
        point["energy_difference_vs_q128"] = point["ground_energy"] - q_reference
    d_reference = basis_scan[-1]["ground_energy"]
    for point in basis_scan:
        point["energy_difference_vs_largest_D"] = point["ground_energy"] - d_reference
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_n4_basis_quadrature_truth_sweep",
        "hamiltonian": "spin-polarized N=4, g=a=omega=1 on R",
        "truth_path": "direct four-index Slater-Condon exterior Hamiltonian",
        "quadrature_scan": quadrature_scan,
        "basis_scan": basis_scan,
        "largest_D_is_a_numerical_reference_not_a_continuum_bound": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Q96-Q128={quadrature_scan[-2]['energy_difference_vs_q128']:.3e} "
        f"D12-D14={basis_scan[-2]['energy_difference_vs_largest_D']:.3e} "
        f"E_D14={d_reference:.12f}"
    )


if __name__ == "__main__":
    main()
