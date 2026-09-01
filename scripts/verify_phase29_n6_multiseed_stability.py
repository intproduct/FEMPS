"""Independently verify the ADR-0019 N=6 stability artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_artifact(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["evidence_level"] != "numerical":
        raise AssertionError("N6 stability must be numerical evidence")
    if data["seeds"] != [31, 37, 43]:
        raise AssertionError("N6 stability seeds disagree with ADR 0019")
    thresholds = data["thresholds"]
    truth = data["dense_ci_audit"]
    points = data["points"]
    point_pass = []
    for index, point in enumerate(points):
        materialization_pass = (
            point["materialized_antisymmetry_residual"]
            <= thresholds["antisymmetry_residual"]
            if index == 0
            else point["materialized_antisymmetry_residual"] is None
        )
        point_pass.append(
            bool(
                point["completed"]
                and -1e-9
                <= point["energy"] - truth["energy"]
                <= thresholds["dense_ci_error"]
                and point["energy_variance"] <= thresholds["variance"]
                and point["norm_error"] <= thresholds["norm_error"]
                and point["structural_antisymmetry_residual"]
                <= thresholds["antisymmetry_residual"]
                and materialization_pass
                and point["retained_rank"] == 4
                and point["retained_condition_number"]
                <= thresholds["retained_condition_number"]
                and point["structural_counts"]["enumerated_virtual_paths"] == 0
                and point["structural_counts"]["materialized_particle_coefficients"]
                == 0
                and abs(point["finite_basis_reference_energy"] - truth["energy"])
                <= thresholds["factorization_error"]
                and point["peak_cpu_rss_bytes"]
                <= thresholds["peak_cpu_rss_bytes"]
                and point["total_elapsed_seconds_this_call"]
                <= thresholds["wall_time_seconds_per_point"]
            )
        )
    spread = max(point["energy"] for point in points) - min(
        point["energy"] for point in points
    )
    operator_pass = bool(
        data["operator_audit"]["backend"] == "physical_operator_svd"
        and data["operator_audit"]["dense_relative_factorization_error"]
        <= thresholds["factorization_error"]
    )
    spread_pass = spread <= thresholds["energy_spread"]
    accepted = all(point_pass) and operator_pass and spread_pass
    if point_pass != data["stability"]["per_run_pass"]:
        raise AssertionError("recorded per-run decisions disagree")
    if data["acceptance"] != {
        "operator_pass": operator_pass,
        "spread_pass": spread_pass,
        "multiseed_pass": accepted,
    }:
        raise AssertionError("recorded N6 stability acceptance disagrees")
    if not accepted:
        raise AssertionError("N6 multiseed stability does not pass ADR 0019")
    return {
        "verified": True,
        "multiseed_pass": True,
        "energy_spread": spread,
        "maximum_dense_ci_error": max(
            point["energy"] - truth["energy"] for point in points
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path("docs/experiments/results/phase29_n6_multiseed_stability.json"),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
