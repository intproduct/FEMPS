"""Independently verify the ADR-0018 N=6 pilot artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_artifact(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["evidence_level"] != "numerical":
        raise AssertionError("N6 pilot must be labeled numerical evidence")
    if data["model"] != {
        "N": 6,
        "D": 10,
        "Q": 128,
        "coupling": 1.0,
        "softening": 1.0,
    }:
        raise AssertionError("N6 pilot model disagrees with ADR 0018")
    thresholds = data["thresholds"]
    truth = data["dense_ci_audit"]
    k1, k4 = data["points"]
    if [point["config"]["terms"] for point in (k1, k4)] != [1, 4]:
        raise AssertionError("expected the registered K=1,4 points")
    operator_pass = bool(
        data["operator_audit"]["backend"] == "physical_operator_svd"
        and data["operator_audit"]["dense_relative_factorization_error"]
        <= thresholds["factorization_error"]
        and all(
            abs(point["finite_basis_reference_energy"] - truth["energy"])
            <= thresholds["factorization_error"]
            for point in (k1, k4)
        )
    )
    truth_pass = bool(
        truth["exterior_dimension"] == 210
        and truth["materialized_particle_coefficients"] == 1_000_000
        and truth["norm_error"] <= thresholds["norm_error"]
        and truth["antisymmetry_residual"] <= thresholds["antisymmetry_residual"]
        and truth["energy_variance"] <= 1e-20
    )
    resources = [
        bool(
            point["peak_cpu_rss_bytes"] <= thresholds["peak_cpu_rss_bytes"]
            and point["total_elapsed_seconds_this_call"]
            <= thresholds["wall_time_seconds_per_point"]
        )
        for point in (k1, k4)
    ]
    state_pass = bool(
        all(point["completed"] for point in (k1, k4))
        and all(point["norm_error"] <= thresholds["norm_error"] for point in (k1, k4))
        and all(
            point["structural_antisymmetry_residual"]
            <= thresholds["antisymmetry_residual"]
            for point in (k1, k4)
        )
        and k4["materialized_antisymmetry_residual"]
        <= thresholds["antisymmetry_residual"]
        and all(
            point["structural_counts"]["enumerated_virtual_paths"] == 0
            for point in (k1, k4)
        )
        and k4["structural_counts"]["materialized_particle_coefficients"] == 0
    )
    k1_error = k1["energy"] - truth["energy"]
    k4_error = k4["energy"] - truth["energy"]
    nested = bool(
        data["diagnostics"]["K4_initial_nested_energy"] <= k1["energy"] + 1e-9
        and k4["energy"] <= data["diagnostics"]["K4_initial_nested_energy"] + 1e-9
    )
    correlation_pass = bool(
        -1e-9 <= k4_error <= thresholds["K4_dense_ci_error"]
        and k4_error <= thresholds["K4_error_ratio_vs_K1"] * k1_error
        and k4["energy_variance"] <= thresholds["K4_variance"]
        and nested
    )
    accepted = bool(
        operator_pass
        and truth_pass
        and all(resources)
        and state_pass
        and correlation_pass
    )
    recomputed = {
        "operator_pass": operator_pass,
        "truth_pass": truth_pass,
        "K1_resource_pass": resources[0],
        "K4_resource_pass": resources[1],
        "state_pass": state_pass,
        "correlation_pass": correlation_pass,
        "pilot_pass": accepted,
    }
    if recomputed != data["acceptance"]:
        raise AssertionError("recorded N6 pilot acceptance disagrees")
    if not accepted:
        raise AssertionError("N6 pilot does not pass ADR 0018")
    return {
        "verified": True,
        "pilot_pass": True,
        "K4_error_ratio_vs_K1": k4_error / k1_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path("docs/experiments/results/phase29_n6_soft_coulomb_pilot.json"),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
