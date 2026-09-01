"""Independently verify the Phase 35 adaptive-pool stability artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from femps.exterior import particle_tt_ranks_exterior_coefficients
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.verify_phase34_adaptive_k_growth import (  # noqa: E402
    _assert_close,
    _complex_vector,
    _tt_storage,
    _verify_growth_record,
)


PARTICLES = 6
DIMENSION = 12
SEED_PAIRS = ((3511, 3512), (3521, 3522), (3531, 3532))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_state(
    point: dict,
    dense_hamiltonian: torch.Tensor,
    ci_energy: float,
) -> dict:
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
        "factorized/dense difference",
    )
    if list(ranks) != point["ordinary_particle_tt_ranks_compact"]:
        raise AssertionError("ordinary particle-TT ranks mismatch")
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
    return {
        "energy": float(energy),
        "variance": float(variance),
        "norm_error": float(abs(norm - 1.0)),
        "ranks": list(ranks),
    }


def verify_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["schema_version"] != 1:
        raise AssertionError("unsupported Phase 35 artifact schema")
    if artifact["evidence_level"] != "numerical":
        raise AssertionError("Phase 35 must remain numerical evidence")
    config = artifact["registered_config"]
    expected_config = {
        "N": 6,
        "D": 12,
        "K_axis": [4, 5, 6],
        "pool_size": 32,
        "seed_pairs": [list(pair) for pair in SEED_PAIRS],
        "steps": 160,
        "lbfgs_steps": 80,
        "device": "cpu",
        "truth_state_initialization": False,
        "truth_constructed_after_all_optimizations": True,
    }
    if config != expected_config:
        raise AssertionError("Phase 35 registered configuration changed")

    sources = artifact["source_records"]
    phase32_path = Path(sources["phase32_artifact"]["path"])
    if _sha256(phase32_path) != sources["phase32_artifact"]["sha256"]:
        raise AssertionError("Phase 32 source artifact hash mismatch")
    phase32 = json.loads(phase32_path.read_text(encoding="utf-8"))
    source_phase32 = next(
        point
        for point in phase32["points"]
        if point["point_id"] == "N6_D12_K4_seed3212_from_D10"
    )
    if source_phase32["checkpoint_sha256"] != sources[
        "phase32_K4_checkpoint_sha256"
    ]:
        raise AssertionError("Phase 32 checkpoint lineage mismatch")
    phase34_path = Path(sources["phase34_historical_control"]["path"])
    if _sha256(phase34_path) != sources["phase34_historical_control"]["sha256"]:
        raise AssertionError("Phase 34 historical artifact hash mismatch")
    phase34 = json.loads(phase34_path.read_text(encoding="utf-8"))
    if not phase34["acceptance"]["phase34_adaptive_growth_pass"]:
        raise AssertionError("Phase 34 historical control is not accepted")

    source_paths = {
        "growth": Path("src/femps/algorithms/diagonal_path_growth.py"),
        "training": Path("src/femps/algorithms/diagonal_path_training.py"),
        "phase34_utilities": Path("scripts/benchmark_phase34_adaptive_k_growth.py"),
        "benchmark": Path("scripts/benchmark_phase35_adaptive_pool_stability.py"),
    }
    for name, source_path in source_paths.items():
        if _sha256(source_path) != sources["source_hashes"][name]:
            raise AssertionError(f"Phase 35 {name} source hash mismatch")

    one_body = harmonic_pair_hamiltonian(
        DIMENSION, kappa=0.0, dtype=torch.complex128, device="cpu"
    )[0]
    dense_pair = soft_coulomb_dense_quadrature(
        DIMENSION,
        quadrature_order=128,
        coupling=1.0,
        softening=1.0,
        dtype=torch.complex128,
        device="cpu",
    )
    dense_pair_check = soft_coulomb_dense_quadrature(
        DIMENSION,
        quadrature_order=160,
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
    _, diagnostics = soft_coulomb_operator(
        DIMENSION,
        quadrature_order=128,
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
    _assert_close(
        artifact["operator"]["direct_ci"]["energy"],
        ci_energy,
        1e-12,
        "CI energy",
    )
    _assert_close(
        artifact["operator"]["direct_ci"]["energy_variance"],
        float(torch.vdot(ci_residual, ci_residual).real),
        1e-18,
        "CI variance",
    )

    source = artifact["source_K4"]
    rebuilt_source = _verify_state(source, dense_hamiltonian, ci_energy)
    torch.testing.assert_close(
        _complex_vector(source["raw_exterior_coefficients"]),
        _complex_vector(source_phase32["raw_exterior_coefficients"]),
        atol=1e-12,
        rtol=1e-12,
        msg="Phase 35 source differs from Phase 32",
    )

    if len(artifact["lineages"]) != 3:
        raise AssertionError("Phase 35 must contain three lineages")
    rebuilt_lineages = []
    all_points = []
    expected_stop_decisions = []
    for index, (lineage, seed_pair) in enumerate(
        zip(artifact["lineages"], SEED_PAIRS, strict=True), start=1
    ):
        if lineage["lineage_id"] != index:
            raise AssertionError("lineage order changed")
        if lineage["seeds"] != {"K5": seed_pair[0], "K6": seed_pair[1]}:
            raise AssertionError("lineage seed pair changed")
        k5 = lineage["K5"]
        k6 = lineage["K6"]
        rebuilt_k5 = _verify_state(k5, dense_hamiltonian, ci_energy)
        rebuilt_k6 = _verify_state(k6, dense_hamiltonian, ci_energy)
        _verify_growth_record(
            lineage["growth"]["K4_to_K5"], source["energy"], k5["energy"]
        )
        _verify_growth_record(
            lineage["growth"]["K5_to_K6"], k5["energy"], k6["energy"]
        )
        realized_k5 = source["dense_quadrature_energy"] - k5[
            "dense_quadrature_energy"
        ]
        realized_k6 = k5["dense_quadrature_energy"] - k6[
            "dense_quadrature_energy"
        ]
        _assert_close(
            lineage["comparison"]["K5_improvement_over_K4"],
            realized_k5,
            1e-13,
            "K5 realized improvement",
        )
        _assert_close(
            lineage["comparison"]["K6_improvement_over_K5"],
            realized_k6,
            1e-13,
            "K6 realized improvement",
        )
        _assert_close(
            lineage["comparison"]["K6_total_improvement_over_K4"],
            realized_k5 + realized_k6,
            1e-13,
            "total realized improvement",
        )
        for transition, growth, realized in (
            ("K4_to_K5", lineage["growth"]["K4_to_K5"], realized_k5),
            ("K5_to_K6", lineage["growth"]["K5_to_K6"], realized_k6),
        ):
            predicted_continue = growth["predicted_improvement"] >= 1e-8
            realized_continue = realized >= 1e-8
            expected_stop_decisions.append(
                {
                    "lineage_id": index,
                    "transition": transition,
                    "predicted_improvement": growth["predicted_improvement"],
                    "realized_improvement": realized,
                    "predicted_decision": (
                        "continue" if predicted_continue else "stop"
                    ),
                    "realized_decision": (
                        "continue" if realized_continue else "stop"
                    ),
                    "agreement": predicted_continue == realized_continue,
                }
            )
        if lineage["pruning_assessment"]["should_prune"]:
            raise AssertionError("unhandled K6 pruning trigger")
        rebuilt_lineages.append(
            {"lineage_id": index, "K5": rebuilt_k5, "K6": rebuilt_k6}
        )
        all_points.extend((k5, k6))

    stopping = artifact["stopping_calibration"]
    if stopping["decisions"] != expected_stop_decisions:
        raise AssertionError("stopping-decision audit mismatch")
    stop_consistent = all(item["agreement"] for item in expected_stop_decisions)
    stop_events = sum(
        item["predicted_decision"] == "stop" for item in expected_stop_decisions
    )
    stop_admitted = stop_consistent and stop_events > 0
    if stopping["all_decisions_consistent"] != stop_consistent:
        raise AssertionError("stopping consistency mismatch")
    if stopping["observed_stop_events"] != stop_events:
        raise AssertionError("stop-event count mismatch")
    if stopping["automatic_stopping_rule_admitted"] != stop_admitted:
        raise AssertionError("automatic stopping admission mismatch")

    cold = artifact["historical_phase34_cold_K6"]
    cold_source = phase34["cold_K6_control"]
    _assert_close(
        cold["energy"],
        cold_source["dense_quadrature_energy"],
        1e-13,
        "historical cold K6 energy",
    )
    thresholds = artifact["thresholds"]
    lineages = artifact["lineages"]
    k6_points = [lineage["K6"] for lineage in lineages]
    monotonic_pass = all(
        lineage["K5"]["dense_quadrature_energy"]
        <= source["dense_quadrature_energy"]
        + thresholds["energy_monotonicity_tolerance"]
        and lineage["K6"]["dense_quadrature_energy"]
        <= lineage["K5"]["dense_quadrature_energy"]
        + thresholds["energy_monotonicity_tolerance"]
        and lineage["comparison"]["K6_total_improvement_over_K4"]
        >= thresholds["minimum_K4_to_K6_improvement"]
        for lineage in lineages
    )
    k6_quality_pass = all(
        point["error_vs_direct_ci"] <= thresholds["K6_CI_error"]
        and point["dense_quadrature_energy_variance"] <= thresholds["K6_variance"]
        for point in k6_points
    )
    k6_energies = [point["dense_quadrature_energy"] for point in k6_points]
    spread = max(k6_energies) - min(k6_energies)
    spread_pass = spread <= thresholds["K6_energy_spread"]
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
        for point in all_points
    )
    truth_pass = all(
        abs(point["factorized_vs_dense_energy_difference"])
        <= thresholds["factorized_vs_dense_energy"]
        for point in all_points
    )
    resource_pass = all(
        point["total_elapsed_seconds_this_call"]
        <= thresholds["wall_time_seconds_per_point"]
        and point["peak_cpu_rss_bytes"] <= thresholds["peak_cpu_rss_bytes"]
        for point in all_points
    )
    operator_pass = (
        artifact["operator"]["physical_operator_svd_relative_error"]
        <= thresholds["operator_factorization_error"]
        and artifact["operator"]["quadrature_relative_change_Q128_vs_Q160"]
        <= thresholds["quadrature_relative_change"]
    )
    pruning_pass = all(
        not lineage["pruning_assessment"]["should_prune"] for lineage in lineages
    )
    recomputed_acceptance = {
        "K_axes_pass": monotonic_pass,
        "K6_quality_pass": k6_quality_pass,
        "K6_spread_pass": spread_pass,
        "structural_pass": structural_pass,
        "truth_reconstruction_pass": truth_pass,
        "operator_pass": operator_pass,
        "resource_pass": resource_pass,
        "pruning_not_triggered_pass": pruning_pass,
        "phase35_pool_stability_pass": bool(
            monotonic_pass
            and k6_quality_pass
            and spread_pass
            and structural_pass
            and truth_pass
            and operator_pass
            and resource_pass
            and pruning_pass
        ),
    }
    if recomputed_acceptance != artifact["acceptance"]:
        raise AssertionError("Phase 35 acceptance record mismatch")
    comparison = artifact["comparison"]
    _assert_close(
        comparison["K6_energy_spread"], spread, 1e-13, "K6 energy spread"
    )
    _assert_close(
        comparison["maximum_K6_error_vs_CI"],
        max(point["error_vs_direct_ci"] for point in k6_points),
        1e-13,
        "maximum K6 CI error",
    )
    return {
        "verified": recomputed_acceptance["phase35_pool_stability_pass"],
        "K6_energies": k6_energies,
        "K6_energy_spread": spread,
        "maximum_K6_error_vs_CI": comparison["maximum_K6_error_vs_CI"],
        "maximum_K6_variance": comparison["maximum_K6_variance"],
        "minimum_K4_to_K6_improvement": comparison[
            "minimum_K4_to_K6_improvement"
        ],
        "automatic_stopping_rule_admitted": stop_admitted,
        "rebuilt_source": rebuilt_source,
        "rebuilt_lineages": rebuilt_lineages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path(
            "docs/experiments/results/phase35_adaptive_pool_stability.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
