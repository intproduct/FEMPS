"""Phase 9 soft-Coulomb quadrature, factorization, and N=2 basis scans."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from femps.basis import harmonic_hamiltonian
from femps.hamiltonians import (
    antisymmetric_two_particle_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
    soft_coulomb_two_fermion_relative_grid_energy,
)


def _ground_energy(
    basis_order: int,
    quadrature_order: int,
    coupling: float,
    softening: float,
    relative_threshold: float,
) -> tuple[float, dict]:
    one_body = harmonic_hamiltonian(basis_order, dtype=torch.complex128)
    interaction, diagnostics = soft_coulomb_operator(
        basis_order,
        quadrature_order=quadrature_order,
        coupling=coupling,
        softening=softening,
        relative_threshold=relative_threshold,
    )
    hamiltonian = antisymmetric_two_particle_hamiltonian(one_body, interaction)
    return float(torch.linalg.eigvalsh(hamiltonian)[0].real), asdict(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--softening", type=float, default=1.0)
    parser.add_argument("--reference-quadrature", type=int, default=128)
    parser.add_argument("--relative-threshold", type=float, default=1e-14)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/soft_coulomb_operator_sweep.json"),
    )
    args = parser.parse_args()

    probe_basis = 8
    reference_dense = soft_coulomb_dense_quadrature(
        probe_basis,
        quadrature_order=args.reference_quadrature,
        coupling=args.coupling,
        softening=args.softening,
    )
    reference_norm = torch.linalg.vector_norm(reference_dense)
    quadrature_scan = []
    for quadrature_order in (24, 32, 48, 64, 96, args.reference_quadrature):
        interaction, diagnostics = soft_coulomb_operator(
            probe_basis,
            quadrature_order=quadrature_order,
            coupling=args.coupling,
            softening=args.softening,
            relative_threshold=args.relative_threshold,
        )
        direct_dense = soft_coulomb_dense_quadrature(
            probe_basis,
            quadrature_order=quadrature_order,
            coupling=args.coupling,
            softening=args.softening,
        )
        quadrature_scan.append(
            {
                "quadrature_order": quadrature_order,
                "direct_quadrature_relative_error_vs_q_reference": float(
                    torch.linalg.vector_norm(direct_dense - reference_dense)
                    / reference_norm
                ),
                "factorized_relative_error_vs_q_reference": float(
                    torch.linalg.vector_norm(interaction.dense() - reference_dense)
                    / reference_norm
                ),
                "diagnostics": asdict(diagnostics),
            }
        )

    threshold_scan = []
    for threshold in (1e-8, 1e-10, 1e-12, 1e-14, 0.0):
        _, diagnostics = soft_coulomb_operator(
            probe_basis,
            quadrature_order=args.reference_quadrature,
            coupling=args.coupling,
            softening=args.softening,
            relative_threshold=threshold,
        )
        threshold_scan.append(asdict(diagnostics))

    basis_scan = []
    for basis_order in (4, 6, 8, 10, 12, 14, 16):
        energy, diagnostics = _ground_energy(
            basis_order,
            args.reference_quadrature,
            args.coupling,
            args.softening,
            args.relative_threshold,
        )
        basis_scan.append(
            {
                "basis_order": basis_order,
                "finite_basis_ground_energy": energy,
                "diagnostics": diagnostics,
            }
        )
    reference_energy = basis_scan[-1]["finite_basis_ground_energy"]
    for point in basis_scan:
        point["energy_difference_vs_d16"] = (
            point["finite_basis_ground_energy"] - reference_energy
        )

    relative_grid_scan = []
    for intervals in (80, 120, 180, 240):
        energy = soft_coulomb_two_fermion_relative_grid_energy(
            intervals=intervals,
            coupling=args.coupling,
            softening=args.softening,
        )
        relative_grid_scan.append(
            {
                "intervals": intervals,
                "half_width": 8.0,
                "spacing": 8.0 / intervals,
                "ground_energy": energy,
            }
        )
    first = relative_grid_scan[-2]
    second = relative_grid_scan[-1]
    h1_squared = first["spacing"] ** 2
    h2_squared = second["spacing"] ** 2
    extrapolated_grid_energy = (
        h1_squared * second["ground_energy"]
        - h2_squared * first["ground_energy"]
    ) / (h1_squared - h2_squared)
    relative_grid_box_scan = []
    for half_width, intervals in ((6.0, 180), (8.0, 240), (10.0, 300)):
        relative_grid_box_scan.append(
            {
                "intervals": intervals,
                "half_width": half_width,
                "spacing": half_width / intervals,
                "ground_energy": soft_coulomb_two_fermion_relative_grid_energy(
                    intervals=intervals,
                    half_width=half_width,
                    coupling=args.coupling,
                    softening=args.softening,
                ),
            }
        )

    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_operator_and_n2_basis_sweep",
        "hamiltonian": {
            "domain": "R",
            "particles": 2,
            "spin_sector": "spin-polarized spinless fermions",
            "trap_frequency": 1.0,
            "coupling": args.coupling,
            "softening": args.softening,
            "pair_potential": "g/sqrt((x_i-x_j)^2+a^2)",
            "units": "hbar=m=omega=1",
        },
        "reference_quadrature_order": args.reference_quadrature,
        "relative_factorization_threshold": args.relative_threshold,
        "quadrature_probe_basis": probe_basis,
        "quadrature_scan": quadrature_scan,
        "factorization_threshold_scan": threshold_scan,
        "basis_scan": basis_scan,
        "independent_relative_coordinate_grid_scan": relative_grid_scan,
        "independent_relative_coordinate_box_scan": relative_grid_box_scan,
        "second_order_grid_extrapolated_energy": extrapolated_grid_energy,
        "d16_minus_grid_extrapolated_energy": (
            reference_energy - extrapolated_grid_energy
        ),
        "largest_basis_energy_is_not_a_proven_continuum_reference": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"q96_relative_error={quadrature_scan[-2]['direct_quadrature_relative_error_vs_q_reference']:.3e} "
        f"E_D16={reference_energy:.12f} "
        f"D14_minus_D16={basis_scan[-2]['energy_difference_vs_d16']:.3e}"
    )


if __name__ == "__main__":
    main()
