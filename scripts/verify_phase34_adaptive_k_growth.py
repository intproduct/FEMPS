"""Independently verify the committed Phase 34 adaptive-growth artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from femps.exterior import particle_tt_ranks_exterior_coefficients
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)


PARTICLES = 6
DIMENSION = 12
QUADRATURE = 128
QUADRATURE_CHECK = 160


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complex_vector(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(
        [complex(real, imaginary) for real, imaginary in values],
        dtype=torch.complex128,
    )


def _tt_storage(dimension: int, ranks: tuple[int, ...]) -> int:
    extended = (1,) + ranks + (1,)
    return sum(
        extended[index] * dimension * extended[index + 1]
        for index in range(len(extended) - 1)
    )


def _assert_close(observed: float, expected: float, tolerance: float, label: str) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(
            f"{label}: observed {observed!r}, expected {expected!r}, "
            f"tolerance {tolerance!r}"
        )


def _verify_growth_record(record: dict, source_energy: float, next_energy: float) -> None:
    if record["truth_state_used"]:
        raise AssertionError("adaptive growth record used a truth state")
    if record["pool_size"] != 32 or len(record["candidates"]) != 32:
        raise AssertionError("adaptive candidate pool does not match registration")
    if [item["candidate_index"] for item in record["candidates"]] != list(range(32)):
        raise AssertionError("adaptive candidate indices are not canonical")
    _assert_close(record["source_energy"], source_energy, 1e-10, "growth source energy")
    admitted = [item for item in record["candidates"] if item["admitted"]]
    if not admitted:
        raise AssertionError("adaptive growth has no admitted candidate")
    selected = min(
        admitted, key=lambda item: (item["predicted_energy"], item["candidate_index"])
    )
    if selected["candidate_index"] != record["selected_candidate"]:
        raise AssertionError("selected candidate is not the registered fixed-span minimum")
    _assert_close(
        record["predicted_energy"],
        selected["predicted_energy"],
        1e-13,
        "selected predicted energy",
    )
    _assert_close(
        record["predicted_improvement"],
        record["source_energy"] - record["predicted_energy"],
        1e-13,
        "selected predicted improvement",
    )
    for candidate in record["candidates"]:
        _assert_close(
            candidate["predicted_improvement"],
            record["source_energy"] - candidate["predicted_energy"],
            1e-13,
            "candidate predicted improvement",
        )
        if candidate["admitted"]:
            if candidate["retained_rank"] != record["source_terms"] + 1:
                raise AssertionError("admitted candidate loses an overlap direction")
            if candidate["retained_condition_number"] > 1e8:
                raise AssertionError("admitted candidate exceeds condition gate")
    if next_energy > record["predicted_energy"] + 1e-9:
        raise AssertionError("nonlinear optimization worsened the selected fixed span")


def verify_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["schema_version"] != 1:
        raise AssertionError("unsupported Phase 34 artifact schema")
    if artifact["evidence_level"] != "numerical":
        raise AssertionError("Phase 34 must remain numerical evidence")
    config = artifact["registered_config"]
    expected_config = {
        "N": 6,
        "D": 12,
        "K_axis": [4, 5, 6],
        "Q": 128,
        "Q_check": 160,
        "pool_size": 32,
        "growth_seeds": {"K5": 3451, "K6": 3452},
        "cold_K6_seed": 3460,
        "steps": 160,
        "lbfgs_steps": 80,
        "learning_rate": 1e-3,
        "final_learning_rate": 1e-5,
        "truth_state_initialization": False,
        "truth_constructed_after_optimization": True,
    }
    if config != expected_config:
        raise AssertionError("Phase 34 registered configuration changed")

    sources = artifact["source_records"]
    phase32_path = Path(sources["phase32_artifact"]["path"])
    if _sha256(phase32_path) != sources["phase32_artifact"]["sha256"]:
        raise AssertionError("Phase 32 source artifact hash mismatch")
    phase32 = json.loads(phase32_path.read_text(encoding="utf-8"))
    if not phase32["acceptance"]["phase32_convergence_pass"]:
        raise AssertionError("Phase 32 source is not accepted")
    phase32_source = next(
        point
        for point in phase32["points"]
        if point["point_id"] == "N6_D12_K4_seed3212_from_D10"
    )
    if (
        phase32_source["checkpoint_sha256"]
        != sources["phase32_K4_checkpoint_sha256"]
    ):
        raise AssertionError("Phase 32 checkpoint lineage hash mismatch")
    source_paths = {
        "growth": Path("src/femps/algorithms/diagonal_path_growth.py"),
        "training": Path("src/femps/algorithms/diagonal_path_training.py"),
        "benchmark": Path("scripts/benchmark_phase34_adaptive_k_growth.py"),
    }
    for name, source_path in source_paths.items():
        if _sha256(source_path) != sources["source_hashes"][name]:
            raise AssertionError(f"Phase 34 {name} source hash mismatch")

    one_body = harmonic_pair_hamiltonian(
        DIMENSION, kappa=0.0, dtype=torch.complex128, device="cpu"
    )[0]
    dense_pair = soft_coulomb_dense_quadrature(
        DIMENSION,
        quadrature_order=QUADRATURE,
        coupling=1.0,
        softening=1.0,
        dtype=torch.complex128,
        device="cpu",
    )
    dense_pair_check = soft_coulomb_dense_quadrature(
        DIMENSION,
        quadrature_order=QUADRATURE_CHECK,
        coupling=1.0,
        softening=1.0,
        dtype=torch.complex128,
        device="cpu",
    )
    quadrature_change = torch.linalg.vector_norm(
        dense_pair - dense_pair_check
    ) / torch.linalg.vector_norm(dense_pair_check)
    _assert_close(
        artifact["operator"]["quadrature_relative_change_Q128_vs_Q160"],
        float(quadrature_change),
        1e-15,
        "quadrature convergence",
    )
    interaction, diagnostics = soft_coulomb_operator(
        DIMENSION,
        quadrature_order=QUADRATURE,
        coupling=1.0,
        softening=1.0,
        relative_threshold=1e-13,
        factorization_backend="physical",
        dtype=torch.complex128,
        device="cpu",
    )
    if diagnostics.retained_rank != artifact["operator"]["physical_operator_svd_rank"]:
        raise AssertionError("operator factor rank mismatch")
    _assert_close(
        diagnostics.dense_relative_factorization_error,
        artifact["operator"]["physical_operator_svd_relative_error"],
        1e-15,
        "operator factorization error",
    )
    dense_hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, PARTICLES, dense_pair
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(dense_hamiltonian)
    ci_energy = float(eigenvalues[0].real)
    ci_coefficients = eigenvectors[:, 0]
    ci_residual = dense_hamiltonian @ ci_coefficients - ci_energy * ci_coefficients
    controls = artifact["operator"]
    _assert_close(controls["direct_ci"]["energy"], ci_energy, 1e-12, "CI energy")
    _assert_close(
        controls["direct_ci"]["energy_variance"],
        float(torch.vdot(ci_residual, ci_residual).real),
        1e-18,
        "CI variance",
    )
    ci_ranks = particle_tt_ranks_exterior_coefficients(
        ci_coefficients, DIMENSION, PARTICLES
    )
    if list(ci_ranks) != controls["direct_ci"]["ordinary_particle_tt_ranks"]:
        raise AssertionError("CI ordinary particle-TT ranks mismatch")

    points = artifact["adaptive_points"] + [artifact["cold_K6_control"]]
    rebuilt: list[dict] = []
    for point in points:
        coefficients = _complex_vector(point["raw_exterior_coefficients"])
        if coefficients.numel() != math.comb(DIMENSION, PARTICLES):
            raise AssertionError("wrong exterior coefficient count")
        norm = torch.vdot(coefficients, coefficients).real
        acted = dense_hamiltonian @ coefficients
        energy = (torch.vdot(coefficients, acted) / norm).real
        residual = acted - energy * coefficients
        variance = torch.vdot(residual, residual).real / norm
        ranks = particle_tt_ranks_exterior_coefficients(
            coefficients, DIMENSION, PARTICLES
        )
        _assert_close(
            point["dense_quadrature_energy"], float(energy), 1e-12, "state energy"
        )
        _assert_close(
            point["dense_quadrature_energy_variance"],
            float(variance),
            1e-12,
            "state variance",
        )
        _assert_close(
            point["dense_quadrature_norm_error"],
            float(abs(norm - 1.0)),
            1e-14,
            "state norm error",
        )
        _assert_close(
            point["error_vs_direct_ci"], float(energy) - ci_energy, 1e-12, "CI error"
        )
        _assert_close(
            point["factorized_vs_dense_energy_difference"],
            point["energy"] - float(energy),
            1e-13,
            "factorized/dense energy difference",
        )
        if list(ranks) != point["ordinary_particle_tt_ranks_compact"]:
            raise AssertionError("state ordinary particle-TT ranks mismatch")
        if _tt_storage(DIMENSION, ranks) != point[
            "ordinary_particle_tt_storage_scalars"
        ]:
            raise AssertionError("ordinary particle-TT storage mismatch")
        terms = point["config"]["terms"]
        if point["femps_stored_parameter_scalars"] != terms * DIMENSION * PARTICLES + terms:
            raise AssertionError("FEMPS storage count mismatch")
        if point["structural_antisymmetry_residual"] != 0.0:
            raise AssertionError("structural antisymmetry residual is nonzero")
        if point["structural_counts"]["enumerated_virtual_paths"] != 0:
            raise AssertionError("production enumerated virtual paths")
        if point["structural_counts"]["materialized_particle_coefficients"] != 0:
            raise AssertionError("production materialized a particle tensor")
        rebuilt.append(
            {
                "id": point["point_id"],
                "energy": float(energy),
                "variance": float(variance),
                "norm_error": float(abs(norm - 1.0)),
                "ranks": list(ranks),
            }
        )

    source, k5, k6, cold = points
    torch.testing.assert_close(
        _complex_vector(source["raw_exterior_coefficients"]),
        _complex_vector(phase32_source["raw_exterior_coefficients"]),
        atol=1e-12,
        rtol=1e-12,
        msg="Phase 34 source coefficients changed from Phase 32",
    )
    _verify_growth_record(
        artifact["growth"]["K4_to_K5"], source["energy"], k5["energy"]
    )
    _verify_growth_record(
        artifact["growth"]["K5_to_K6"], k5["energy"], k6["energy"]
    )
    thresholds = artifact["thresholds"]
    monotonic_pass = (
        k5["dense_quadrature_energy"]
        <= source["dense_quadrature_energy"]
        + thresholds["energy_monotonicity_tolerance"]
        and k6["dense_quadrature_energy"]
        <= k5["dense_quadrature_energy"]
        + thresholds["energy_monotonicity_tolerance"]
    )
    structural_pass = all(
        point["completed"]
        and point["norm_error"] <= thresholds["norm_error"]
        and point["dense_quadrature_norm_error"] <= thresholds["norm_error"]
        and point["structural_antisymmetry_residual"]
        <= thresholds["antisymmetry_residual"]
        and point["retained_rank"] == point["config"]["terms"]
        and point["retained_condition_number"]
        <= thresholds["retained_condition_number"]
        and point["structural_counts"]["enumerated_virtual_paths"] == 0
        and point["structural_counts"]["materialized_particle_coefficients"] == 0
        for point in (k5, k6, cold)
    )
    truth_pass = all(
        abs(point["factorized_vs_dense_energy_difference"])
        <= thresholds["factorized_vs_dense_energy"]
        for point in (k5, k6, cold)
    )
    resource_pass = all(
        point["total_elapsed_seconds_this_call"]
        <= thresholds["wall_time_seconds_per_point"]
        and point["peak_cpu_rss_bytes"] <= thresholds["peak_cpu_rss_bytes"]
        for point in (k5, k6, cold)
    )
    operator_pass = (
        artifact["operator"]["physical_operator_svd_relative_error"]
        <= thresholds["operator_factorization_error"]
        and artifact["operator"]["quadrature_relative_change_Q128_vs_Q160"]
        <= thresholds["quadrature_relative_change"]
    )
    selection_pass = all(
        record["predicted_energy"] <= record["source_energy"] + 1e-10
        and record["candidates"][record["selected_candidate"]]["admitted"]
        for record in artifact["growth"].values()
    )
    pruning_pass = not artifact["pruning_assessment"]["should_prune"]
    recomputed_acceptance = {
        "selection_pass": selection_pass,
        "K_axis_monotonic_pass": monotonic_pass,
        "structural_pass": structural_pass,
        "truth_reconstruction_pass": truth_pass,
        "operator_pass": operator_pass,
        "resource_pass": resource_pass,
        "pruning_not_triggered_pass": pruning_pass,
        "phase34_adaptive_growth_pass": bool(
            selection_pass
            and monotonic_pass
            and structural_pass
            and truth_pass
            and operator_pass
            and resource_pass
            and pruning_pass
        ),
    }
    if recomputed_acceptance != artifact["acceptance"]:
        raise AssertionError("Phase 34 acceptance record mismatch")

    comparison = artifact["comparison"]
    _assert_close(
        comparison["K5_energy_improvement"],
        source["dense_quadrature_energy"] - k5["dense_quadrature_energy"],
        1e-13,
        "K5 improvement",
    )
    _assert_close(
        comparison["K6_energy_improvement_over_K5"],
        k5["dense_quadrature_energy"] - k6["dense_quadrature_energy"],
        1e-13,
        "K6 incremental improvement",
    )
    _assert_close(
        comparison["adaptive_K6_minus_cold_K6_energy"],
        k6["dense_quadrature_energy"] - cold["dense_quadrature_energy"],
        1e-13,
        "adaptive/cold K6 comparison",
    )
    measurable = (
        comparison["K6_total_energy_improvement_over_K4"]
        >= thresholds["measurable_improvement"]
    )
    if comparison["adaptive_growth_measurable"] != measurable:
        raise AssertionError("measurable-improvement decision mismatch")
    return {
        "verified": recomputed_acceptance["phase34_adaptive_growth_pass"],
        "K4_energy": source["dense_quadrature_energy"],
        "K5_energy": k5["dense_quadrature_energy"],
        "K6_energy": k6["dense_quadrature_energy"],
        "cold_K6_energy": cold["dense_quadrature_energy"],
        "K6_error_vs_CI": k6["error_vs_direct_ci"],
        "K6_variance": k6["dense_quadrature_energy_variance"],
        "rebuilt_points": rebuilt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path("docs/experiments/results/phase34_adaptive_k_growth.json"),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
