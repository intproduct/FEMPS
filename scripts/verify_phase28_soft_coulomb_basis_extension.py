"""Independently verify the recorded D8-D12 soft-Coulomb lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_artifact(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["evidence_level"] != "numerical":
        raise AssertionError("basis extension must be labeled numerical evidence")
    thresholds = data["thresholds"]
    points = data["extension_points"]
    if [p["config"]["basis_order"] for p in points] != [10, 12]:
        raise AssertionError("expected the registered D10 and D12 points")

    def passes(point: dict) -> bool:
        return bool(
            point["completed"]
            and -1e-9 <= point["error_vs_dense_quadrature_ci"] <= thresholds["dense_ci_error"]
            and point["energy_variance"] <= thresholds["variance"]
            and point["norm_error"] <= thresholds["norm_error"]
            and point["structural_antisymmetry_residual"] <= thresholds["antisymmetry_residual"]
            and point["materialized_antisymmetry_residual"] <= thresholds["antisymmetry_residual"]
            and point["structural_counts"]["enumerated_virtual_paths"] == 0
            and point["peak_cpu_rss_bytes"] > 0
        )

    point_pass = [passes(point) for point in points]
    operator_pass = all(
        audit["operator"]["backend"] == "physical_operator_svd"
        and audit["operator"]["dense_relative_factorization_error"] <= thresholds["factorization_error"]
        for audit in data["operator_and_truth_audits"]
    )
    axis = data["convergence"]["D_axis_K4"]
    if [p["D"] for p in axis] != [8, 10, 12]:
        raise AssertionError("basis axis must be D=8,10,12")
    monotone = all(
        b["absolute_error_vs_D14"] <= a["absolute_error_vs_D14"] + 1e-9
        for a, b in zip(axis, axis[1:])
    )
    accepted = all(point_pass) and operator_pass and monotone
    if accepted != data["acceptance"]["basis_extension_pass"]:
        raise AssertionError("recorded basis-extension acceptance disagrees")
    if not accepted:
        raise AssertionError("basis-extension acceptance does not pass")
    return {"verified": True, "basis_extension_pass": True, "dimensions": [8, 10, 12]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path("docs/experiments/results/phase28_soft_coulomb_basis_extension.json"),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
