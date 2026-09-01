"""Independent verifier for the exploratory Phase 39 correlated carrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from femps.algorithms.correlated_exterior import (
    correlated_two_fermion_observables,
    project_correlated_two_fermion_coefficients,
)
from femps.basis import harmonic_hamiltonian
from femps.hamiltonians import (
    antisymmetric_two_particle_hamiltonian,
    soft_coulomb_operator,
)


DEFAULT_INPUT = Path(
    "docs/experiments/results/phase39_correlated_carrier_prototype.json"
)


def _carrier(order: int) -> torch.Tensor:
    value = torch.zeros((order, 2), dtype=torch.float64)
    value[0, 0] = 1.0
    value[1, 1] = 1.0
    return value


def _ci_energy(order: int) -> float:
    one_body = harmonic_hamiltonian(order, dtype=torch.complex128)
    interaction, _ = soft_coulomb_operator(
        order,
        quadrature_order=128,
        coupling=1.0,
        softening=1.0,
        relative_threshold=0.0,
        factorization_backend="physical",
        dtype=torch.complex128,
    )
    hamiltonian = antisymmetric_two_particle_hamiltonian(one_body, interaction)
    return float(torch.linalg.eigvalsh(hamiltonian)[0].real)


def verify(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("unsupported artifact schema")
    if record.get("evidence_level") != "exploratory numerical evidence":
        raise ValueError("prototype must remain exploratory numerical evidence")
    if "not a preregistered benchmark" not in record["scientific_boundary"]:
        raise ValueError("prototype boundary is missing")

    amplitudes = torch.tensor(
        record["optimization"]["optimized_amplitudes"], dtype=torch.float64
    )
    exponents = torch.tensor(
        record["model"]["correlator_exponents"], dtype=torch.float64
    )
    carrier = _carrier(int(record["model"]["carrier_basis_order"]))
    result = correlated_two_fermion_observables(
        carrier,
        amplitudes,
        exponents,
        quadrature_order=160,
        coupling=1.0,
        softening=1.0,
    )
    final = record["final"]
    checks = {
        "energy_absolute_difference": abs(float(result.energy) - final["energy"]),
        "norm_absolute_difference": abs(float(result.norm) - final["raw_norm"]),
        "variance_absolute_difference": abs(
            float(result.energy_variance) - final["energy_variance"]
        ),
        "antisymmetry_residual": float(result.antisymmetry_residual),
        "correlator_symmetry_residual": float(result.correlator_symmetry_residual),
    }
    if max(
        checks["energy_absolute_difference"],
        checks["norm_absolute_difference"],
        checks["variance_absolute_difference"],
    ) > 1e-12:
        raise AssertionError("serialized correlated-carrier observables do not reproduce")
    if max(
        checks["antisymmetry_residual"], checks["correlator_symmetry_residual"]
    ) > 1e-13:
        raise AssertionError("symmetry residual exceeds the bounded tolerance")

    reconstructed_projection = []
    for stored in record["projection_rank_axis"]:
        order = int(stored["projection_order"])
        coefficients = project_correlated_two_fermion_coefficients(
            carrier,
            amplitudes,
            exponents,
            projection_order=order,
            quadrature_order=160,
        ).detach()
        singular_values = torch.linalg.svdvals(coefficients)
        matrix_rank = int(
            torch.count_nonzero(singular_values > 1e-10 * singular_values[0])
        )
        if matrix_rank != stored["antisymmetric_matrix_rank"]:
            raise AssertionError(f"projection rank mismatch at D={order}")
        skew = float(
            torch.linalg.vector_norm(coefficients + coefficients.mT)
            / torch.linalg.vector_norm(coefficients)
        )
        if skew > 1e-13:
            raise AssertionError(f"projection skew residual failed at D={order}")
        reconstructed_projection.append(
            {"projection_order": order, "matrix_rank": matrix_rank, "skew": skew}
        )

    ci_differences = []
    for stored in record["same_basis_ci_axis"]:
        order = int(stored["basis_order"])
        difference = abs(_ci_energy(order) - stored["same_basis_ci_energy"])
        if difference > 1e-11:
            raise AssertionError(f"same-basis CI mismatch at D={order}")
        ci_differences.append({"basis_order": order, "absolute_difference": difference})

    if record["resources"]["production_virtual_path_enumeration"] != 0:
        raise AssertionError("prototype must not claim production path enumeration")
    if not record["resources"]["bounded_coordinate_grid_materialized"]:
        raise AssertionError("bounded Q^2 materialization must remain explicit")
    return {
        "verified": True,
        "checks": checks,
        "projection_axis": reconstructed_projection,
        "ci_axis": ci_differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.input), indent=2))


if __name__ == "__main__":
    main()
