"""Independently verify the recorded Phase 28 E4 acceptance artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FINITE_ERROR_LIMIT = 1e-3
VARIANCE_LIMIT = 1e-2
ANTISYMMETRY_LIMIT = 1e-12


def _passes_stability(point: dict) -> bool:
    materialized = point["materialized_antisymmetry_residual"]
    return bool(
        point["completed"]
        and -1e-9 <= point["error_vs_finite_basis"] <= FINITE_ERROR_LIMIT
        and point["energy_variance"] <= VARIANCE_LIMIT
        and point["norm_error"] <= 1e-10
        and point["structural_antisymmetry_residual"] <= ANTISYMMETRY_LIMIT
        and (materialized is None or materialized <= ANTISYMMETRY_LIMIT)
        and point["polynomial_explicit_absolute_difference"] <= 1e-10
        and point["structural_counts"]["enumerated_virtual_paths"] == 0
        and point["structural_counts"]["materialized_particle_coefficients"] == 0
        and point["peak_cpu_rss_bytes"] > 0
        and point["cpu_memory"]["samples"] >= 2
    )


def verify_artifact(payload: dict) -> dict:
    if payload.get("evidence_level") != "numerical":
        raise AssertionError("E4 artifact must be labeled numerical evidence")
    d6 = payload["d6_blind_multiseed"]
    d7 = payload["d7_nested_basis_continuation_multiseed"]
    if len(d6) < 3 or len(d7) < 3:
        raise AssertionError("E4 stability requires at least three runs per group")
    if not all(_passes_stability(point) for point in d6 + d7):
        raise AssertionError("one or more E4 stability runs fail the raw criteria")
    if any(
        point["initialization_lineage"]["truth_state_used"]
        for point in d6 + d7
    ):
        raise AssertionError("truth-state initialization is forbidden")
    k_axis = payload["convergence"]["K_axis_D6"]
    if [point["K"] for point in k_axis] != [1, 2, 4]:
        raise AssertionError("K-axis points must be K=1,2,4")
    if any(
        right["energy"] > left["energy"] + 1e-9
        for left, right in zip(k_axis, k_axis[1:])
    ):
        raise AssertionError("K-axis energy is not nonincreasing")
    d_axis = payload["convergence"]["D_axis_K4"]
    if [point["D"] for point in d_axis] != [5, 6, 7]:
        raise AssertionError("D-axis points must be D=5,6,7")
    if any(
        right["absolute_continuum_error"]
        > left["absolute_continuum_error"] + 1e-9
        for left, right in zip(d_axis, d_axis[1:])
    ):
        raise AssertionError("D-axis continuum error is not nonincreasing")
    comparators = payload["comparators"]
    for name in ("single_slater_D6", "exact_ci_D6", "exact_ci_D7", "single_agp_D6"):
        if name not in comparators:
            raise AssertionError(f"missing comparator: {name}")
    if comparators["exact_ci_D6"]["ordinary_particle_tt_ranks"] != [6, 15, 6]:
        raise AssertionError("unexpected D6 exact-CI particle-TT ranks")
    if comparators["exact_ci_D7"]["ordinary_particle_tt_ranks"] != [7, 21, 7]:
        raise AssertionError("unexpected D7 exact-CI particle-TT ranks")
    acceptance = payload["acceptance"]
    if not all(
        acceptance[key]
        for key in (
            "E4_pass",
            "cpu_peak_memory_complete",
            "no_virtual_path_enumeration",
            "all_structural_antisymmetry_residuals_within_tolerance",
        )
    ):
        raise AssertionError("recorded acceptance flags are not all true")
    return {
        "verified": True,
        "evidence_level": "numerical",
        "D6_runs": len(d6),
        "D7_runs": len(d7),
        "maximum_D6_finite_basis_error": max(
            point["error_vs_finite_basis"] for point in d6
        ),
        "maximum_D7_finite_basis_error": max(
            point["error_vs_finite_basis"] for point in d7
        ),
        "maximum_D7_variance": max(point["energy_variance"] for point in d7),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path("docs/experiments/results/phase28_e4_closure.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    print(json.dumps(verify_artifact(payload), indent=2))


if __name__ == "__main__":
    main()
