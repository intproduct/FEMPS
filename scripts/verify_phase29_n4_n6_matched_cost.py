"""Verify the fixed-D,K,L N4-to-N6 transition-cost artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_artifact(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["evidence_level"] != "numerical":
        raise AssertionError("matched cost audit must be numerical evidence")
    points = data["points"]
    if [(point["N"], point["D"], point["K"], point["L"]) for point in points] != [
        (4, 10, 4, 19),
        (6, 10, 4, 19),
    ]:
        raise AssertionError("matched cost axes disagree")
    point_pass = [
        bool(
            point["auto_minor_overlap_max_absolute_difference"] <= 1e-10
            and point["auto_minor_hamiltonian_max_absolute_difference"] <= 1e-10
            and point["structural_counts"]["enumerated_virtual_paths"] == 0
            and point["structural_counts"]["materialized_particle_coefficients"]
            == 0
            and all(
                mode["median_seconds"] > 0 and mode["cpu_memory"]["peak_rss_bytes"] > 0
                for mode in point["modes"].values()
            )
        )
        for point in points
    ]
    operator_pass = bool(
        data["operator_audit"]["backend"] == "physical_operator_svd"
        and data["operator_audit"]["rank"] == 19
        and data["operator_audit"]["dense_relative_factorization_error"] <= 1e-11
    )
    accepted = all(point_pass) and operator_pass
    if data["acceptance"] != {
        "per_point_pass": point_pass,
        "operator_pass": operator_pass,
        "matched_cost_audit_pass": accepted,
    }:
        raise AssertionError("recorded matched-cost acceptance disagrees")
    ratios = data["N6_over_N4_ratios"]
    if ratios["stored_orbital_scalars_N6_over_N4"] != 1.5:
        raise AssertionError("stored-state ratio disagrees with KDN")
    if ratios["one_body_determinants_N6_over_N4"] != 1.5:
        raise AssertionError("one-body count ratio disagrees with K^2 N")
    if ratios["two_body_determinants_N6_over_N4"] != 2.5:
        raise AssertionError("two-body count ratio disagrees with K^2 L N(N-1)")
    if not accepted:
        raise AssertionError("matched N4-to-N6 cost audit does not pass")
    return {"verified": True, "matched_cost_audit_pass": True, "ratios": ratios}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path("docs/experiments/results/phase29_n4_n6_matched_cost.json"),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
