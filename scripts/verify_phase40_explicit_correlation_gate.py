"""Independent reconstruction of the Phase 40 differentiator artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from femps.algorithms import canonical_two_orbital_carrier
from femps.algorithms.correlated_exterior import correlated_two_fermion_observables
from femps.basis import harmonic_hamiltonian
from femps.exterior import (
    diagonal_path_energy,
    diagonal_path_exterior_coefficients,
    diagonal_path_norm,
)
from femps.hamiltonians import (
    antisymmetric_two_particle_hamiltonian,
    soft_coulomb_operator,
    soft_coulomb_two_fermion_relative_grid_energy,
)


DEFAULT_INPUT = Path("docs/experiments/results/phase40_explicit_correlation_gate.json")
EXPECTED_D = [2, 4, 6, 8, 10, 12]
EXPECTED_P = {"0": [], "1": [1.0], "3": [0.25, 1.0, 4.0], "5": [0.0625, 0.25, 1.0, 4.0, 16.0]}
EXPECTED_K = [1, 2, 4]
EXPECTED_CORRELATED_SEEDS = [40001, 40002, 40003]
EXPECTED_NOCI_SEEDS = [40101, 40102, 40103]


def _sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _complex_tensor(record: dict[str, Any]) -> torch.Tensor:
    return torch.complex(
        torch.tensor(record["real"], dtype=torch.float64),
        torch.tensor(record["imag"], dtype=torch.float64),
    )


def _operators(dimension: int):
    one_body = harmonic_hamiltonian(dimension, dtype=torch.complex128)
    interaction, _ = soft_coulomb_operator(
        dimension,
        quadrature_order=128,
        coupling=1.0,
        softening=1.0,
        relative_threshold=0.0,
        factorization_backend="physical",
        dtype=torch.complex128,
        device="cpu",
    )
    return one_body, interaction


def _gradient_errors(point: dict[str, Any]) -> tuple[float, float]:
    raw = torch.tensor(point["raw_carrier"], dtype=torch.float64, requires_grad=True)
    amplitudes = torch.tensor(
        point["amplitudes"], dtype=torch.float64, requires_grad=True
    )
    exponents = torch.tensor(point["exponents"], dtype=torch.float64)

    def energy(raw_value: torch.Tensor, amplitude_value: torch.Tensor) -> torch.Tensor:
        return correlated_two_fermion_observables(
            canonical_two_orbital_carrier(raw_value),
            amplitude_value,
            exponents,
            quadrature_order=96,
            coupling=1.0,
            softening=1.0,
        ).energy

    value = energy(raw, amplitudes)
    gradients = torch.autograd.grad(value, (raw, amplitudes), allow_unused=True)
    row = 2 if raw.shape[0] > 2 else 0
    step = 1e-5
    plus_raw = raw.detach().clone()
    minus_raw = raw.detach().clone()
    plus_raw[row, 0] += step
    minus_raw[row, 0] -= step
    raw_fd = (
        energy(plus_raw, amplitudes.detach())
        - energy(minus_raw, amplitudes.detach())
    ) / (2.0 * step)
    raw_ad = gradients[0][row, 0] if gradients[0] is not None else raw_fd.new_zeros(())
    raw_error = float(torch.abs(raw_ad - raw_fd))
    amplitude_error = 0.0
    if amplitudes.numel():
        plus_amplitudes = amplitudes.detach().clone()
        minus_amplitudes = amplitudes.detach().clone()
        plus_amplitudes[0] += step
        minus_amplitudes[0] -= step
        amplitude_fd = (
            energy(raw.detach(), plus_amplitudes)
            - energy(raw.detach(), minus_amplitudes)
        ) / (2.0 * step)
        amplitude_ad = (
            gradients[1][0] if gradients[1] is not None else amplitude_fd.new_zeros(())
        )
        amplitude_error = float(torch.abs(amplitude_ad - amplitude_fd))
    return raw_error, amplitude_error


def _verify_frozen_config(record: dict[str, Any]) -> None:
    frozen = record["frozen_config"]
    expected = {
        "D_axis": EXPECTED_D,
        "P_features": EXPECTED_P,
        "K_axis": EXPECTED_K,
        "correlated_seeds": EXPECTED_CORRELATED_SEEDS,
        "noci_seeds": EXPECTED_NOCI_SEEDS,
        "optimization_quadrature": 96,
        "audit_quadrature": [128, 160],
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise AssertionError(f"frozen Phase 40 config mismatch: {key}")
    thresholds = frozen["thresholds"]
    if thresholds != {
        "energy_q128_q160": 1e-7,
        "relative_norm_q128_q160": 1e-8,
        "antisymmetry": 1e-12,
        "gradient_absolute_difference": 1e-6,
        "norm": 1e-10,
        "explicit_contraction": 1e-10,
        "primary_error_ratio": 0.5,
        "minimum_passing_seeds": 2,
        "minimum_consecutive_D": 2,
    }:
        raise AssertionError("frozen Phase 40 thresholds changed")


def verify(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("status") != "complete":
        raise ValueError("Phase 40 artifact is absent or incomplete")
    if record.get("evidence_level") != "preregistered numerical evidence":
        raise ValueError("Phase 40 evidence label changed")
    boundary = record.get("scientific_boundary", "")
    if "not a scalable contraction result" not in boundary or "Paper B" not in boundary:
        raise ValueError("Phase 40 scientific boundary is incomplete")
    _verify_frozen_config(record)
    for source, digest in record["source_hashes"].items():
        if _sha256(Path(source)) != digest:
            raise AssertionError(f"source hash mismatch: {source}")

    reference = soft_coulomb_two_fermion_relative_grid_energy(
        intervals=2000,
        half_width=10.0,
        coupling=1.0,
        softening=1.0,
    )
    if abs(reference - record["reference"]["continuum_grid_energy"]) > 1e-12:
        raise AssertionError("independent relative-coordinate reference mismatch")

    correlated = record["correlated_points"]
    noci = record["noci_points"]
    if len(correlated) != 72 or len(noci) != 54:
        raise AssertionError("Phase 40 production axes are incomplete")
    correlated_keys = {(p["D"], p["P"], p["seed"]) for p in correlated}
    expected_correlated_keys = {
        (dimension, feature_count, seed)
        for dimension in EXPECTED_D
        for feature_count in (0, 1, 3, 5)
        for seed in EXPECTED_CORRELATED_SEEDS
    }
    if correlated_keys != expected_correlated_keys:
        raise AssertionError("correlated D/P/seed Cartesian product mismatch")
    noci_keys = {(p["D"], p["K"], p["seed"]) for p in noci}
    expected_noci_keys = {
        (dimension, terms, seed)
        for dimension in EXPECTED_D
        for terms in EXPECTED_K
        for seed in EXPECTED_NOCI_SEEDS
    }
    if noci_keys != expected_noci_keys:
        raise AssertionError("NOCI D/K/seed Cartesian product mismatch")

    maximum_correlated_difference = 0.0
    maximum_gradient_difference = 0.0
    reconstructed_correlated_validation: dict[tuple[int, int, int], bool] = {}
    for point in correlated:
        orbitals = torch.tensor(point["orbitals"], dtype=torch.float64)
        amplitudes = torch.tensor(point["amplitudes"], dtype=torch.float64)
        exponents = torch.tensor(point["exponents"], dtype=torch.float64)
        q128 = correlated_two_fermion_observables(
            orbitals,
            amplitudes,
            exponents,
            quadrature_order=128,
            coupling=1.0,
            softening=1.0,
        )
        q160 = correlated_two_fermion_observables(
            orbitals,
            amplitudes,
            exponents,
            quadrature_order=160,
            coupling=1.0,
            softening=1.0,
        )
        differences = (
            abs(float(q160.energy) - point["energy"]),
            abs(float(q160.norm) - point["raw_norm"]),
            abs(float(q160.energy_variance) - point["energy_variance"]),
        )
        maximum_correlated_difference = max(maximum_correlated_difference, *differences)
        if max(differences) > 2e-11:
            raise AssertionError(
                f"correlated reconstruction mismatch at D/P/seed={point['D']}/{point['P']}/{point['seed']}"
            )
        energy_change = abs(float(q160.energy) - float(q128.energy))
        norm_change = float(torch.abs(q160.norm - q128.norm) / torch.abs(q160.norm))
        raw_gradient_error, amplitude_gradient_error = _gradient_errors(point)
        maximum_gradient_difference = max(
            maximum_gradient_difference, raw_gradient_error, amplitude_gradient_error
        )
        validation = (
            float(q160.antisymmetry_residual) <= 1e-12
            and float(q160.correlator_symmetry_residual) <= 1e-12
            and energy_change <= 1e-7
            and norm_change <= 1e-8
            and raw_gradient_error <= 1e-6
            and amplitude_gradient_error <= 1e-6
        )
        reconstructed_correlated_validation[(point["D"], point["P"], point["seed"])] = validation
        if validation != point["validation"]["all_pass"]:
            raise AssertionError("serialized correlated validation flag mismatch")

    operator_cache = {dimension: _operators(dimension) for dimension in EXPECTED_D}
    maximum_noci_energy_difference = 0.0
    reconstructed_noci_validation: dict[tuple[int, int, int], bool] = {}
    for point in noci:
        orbitals = _complex_tensor(point["orbitals"])
        amplitudes = _complex_tensor(point["linear_amplitudes"])
        one_body, interaction = operator_cache[point["D"]]
        energy = diagonal_path_energy(
            orbitals,
            amplitudes,
            one_body,
            two_body_left=interaction.left,
            two_body_right=interaction.right,
            two_body_weights=interaction.weights,
        )
        norm = diagonal_path_norm(orbitals, amplitudes)
        coefficients = diagonal_path_exterior_coefficients(orbitals, amplitudes)
        hamiltonian = antisymmetric_two_particle_hamiltonian(one_body, interaction)
        acted = hamiltonian @ coefficients
        variance = torch.vdot(
            acted - energy * coefficients, acted - energy * coefficients
        ).real / torch.vdot(coefficients, coefficients).real
        differences = (
            abs(float(energy) - point["energy"]),
            abs(float(norm) - point["norm"]),
            abs(float(variance) - point["energy_variance"]),
        )
        maximum_noci_energy_difference = max(
            maximum_noci_energy_difference, *differences
        )
        if max(differences) > 2e-10:
            raise AssertionError(
                f"NOCI reconstruction mismatch at D/K/seed={point['D']}/{point['K']}/{point['seed']}"
            )
        validation = (
            point["norm_error"] <= 1e-10
            and point["structural_antisymmetry_residual"] <= 1e-12
            and point["materialized_antisymmetry_residual"] <= 1e-12
            and point["polynomial_explicit_absolute_difference"] <= 1e-10
            and point["materialization"]["virtual_paths"] == 0
        )
        reconstructed_noci_validation[(point["D"], point["K"], point["seed"])] = validation
        if validation != point["validation"]["all_pass"]:
            raise AssertionError("serialized NOCI validation flag mismatch")

    for ci in record["same_basis_ci"]:
        one_body, interaction = operator_cache[ci["D"]]
        hamiltonian = antisymmetric_two_particle_hamiltonian(one_body, interaction)
        energy = float(torch.linalg.eigvalsh(hamiltonian)[0].real)
        if abs(energy - ci["energy"]) > 1e-11:
            raise AssertionError(f"same-basis CI mismatch at D={ci['D']}")

    reconstructed_axis = []
    for dimension in EXPECTED_D:
        corr_seeds = [
            point for point in correlated if point["D"] == dimension and point["P"] == 5
        ]
        noci_seeds = [
            point for point in noci if point["D"] == dimension and point["K"] == 4
        ]
        best_corr = min(corr_seeds, key=lambda point: point["energy"])
        best_noci = min(noci_seeds, key=lambda point: point["energy"])
        noci_error = abs(best_noci["energy"] - reference)
        seed_passes = [
            abs(point["energy"] - reference)
            + point["energy_uncertainty_q128_q160"]
            <= 0.5 * noci_error
            and reconstructed_correlated_validation[(dimension, 5, point["seed"])]
            for point in corr_seeds
        ]
        point_pass = (
            abs(best_corr["energy"] - reference)
            + best_corr["energy_uncertainty_q128_q160"]
            <= 0.5 * noci_error
            and best_corr["optimized_real_parameter_count"]
            <= best_noci["optimized_real_parameter_count"]
            and sum(seed_passes) >= 2
            and reconstructed_noci_validation[(dimension, 4, best_noci["seed"])]
        )
        reconstructed_axis.append({"D": dimension, "point_pass": point_pass})
    consecutive = [
        [left["D"], right["D"]]
        for left, right in zip(reconstructed_axis, reconstructed_axis[1:])
        if left["point_pass"] and right["point_pass"]
    ]
    accepted = (
        all(reconstructed_correlated_validation.values())
        and all(reconstructed_noci_validation.values())
        and bool(consecutive)
    )
    if consecutive != record["comparison"]["consecutive_advantage_pairs"]:
        raise AssertionError("consecutive-D gate reconstruction mismatch")
    if accepted != record["acceptance"]["phase40_differentiator_pass"]:
        raise AssertionError("Phase 40 acceptance reconstruction mismatch")
    return {
        "verified": True,
        "phase40_differentiator_pass": accepted,
        "correlated_points_reconstructed": len(correlated),
        "noci_points_reconstructed": len(noci),
        "maximum_correlated_observable_difference": maximum_correlated_difference,
        "maximum_noci_observable_difference": maximum_noci_energy_difference,
        "maximum_gradient_absolute_difference": maximum_gradient_difference,
        "consecutive_advantage_pairs": consecutive,
        "independent_reproduction_still_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.input), indent=2))


if __name__ == "__main__":
    main()
