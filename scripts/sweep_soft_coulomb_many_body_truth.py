"""Direct exterior truth scans for N>4 soft-Coulomb benchmarks."""

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


def _point(particles: int, dimension: int, quadrature: int) -> dict:
    started = time.perf_counter()
    one_body = harmonic_hamiltonian(dimension, dtype=torch.complex128)
    interaction = soft_coulomb_dense_quadrature(
        dimension, quadrature_order=quadrature
    )
    hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, particles, interaction
    )
    return {
        "N": particles,
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
    parser.add_argument("--particles", type=int, required=True)
    parser.add_argument("--basis-orders", nargs="+", type=int, required=True)
    parser.add_argument("--quadrature-orders", nargs="+", type=int, default=[96, 128, 160])
    parser.add_argument("--production-quadrature", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.particles % 2:
        raise ValueError("this fixed-number AGP benchmark requires even N")
    largest_basis = max(args.basis_orders)
    basis_scan = [
        _point(args.particles, dimension, args.production_quadrature)
        for dimension in args.basis_orders
    ]
    quadrature_scan = [
        _point(args.particles, largest_basis, quadrature)
        for quadrature in args.quadrature_orders
        if quadrature != args.production_quadrature
    ]
    production_point = next(
        point for point in basis_scan if point["D"] == largest_basis
    )
    quadrature_scan.append(production_point)
    quadrature_scan.sort(key=lambda point: point["Q"])
    largest_q_energy = quadrature_scan[-1]["ground_energy"]
    for point in quadrature_scan:
        point["energy_difference_vs_largest_Q"] = (
            point["ground_energy"] - largest_q_energy
        )
    largest_d_energy = basis_scan[-1]["ground_energy"]
    for point in basis_scan:
        point["energy_difference_vs_largest_D"] = (
            point["ground_energy"] - largest_d_energy
        )
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_many_body_direct_exterior_truth",
        "hamiltonian": (
            f"spin-polarized N={args.particles}, g=a=omega=1 on R"
        ),
        "truth_path": "direct four-index Slater-Condon exterior Hamiltonian",
        "largest_D_and_Q_are_numerical_references_not_continuum_truth": True,
        "basis_scan": basis_scan,
        "quadrature_scan_at_largest_D": quadrature_scan,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"N={args.particles} D={largest_basis} "
        f"Q={args.production_quadrature}-Q={max(args.quadrature_orders)}: "
        f"{production_point['ground_energy'] - largest_q_energy:.3e}"
    )


if __name__ == "__main__":
    main()
