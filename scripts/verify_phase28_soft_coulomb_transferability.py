"""Independently recompute acceptance from the soft-Coulomb artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_artifact(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["evidence_level"] != "numerical":
        raise AssertionError("soft-Coulomb artifact must be numerical evidence")
    thresholds = data["thresholds"]
    all_points = data["d6_blind_multiseed"] + data["d8_nested_basis_continuation_multiseed"] + data["axis_points"]
    if len(data["d6_blind_multiseed"]) < 3 or len(data["d8_nested_basis_continuation_multiseed"]) < 3:
        raise AssertionError("stability requires three runs per D group")

    def point_pass(point: dict) -> bool:
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

    stability = all(point_pass(p) for p in data["d6_blind_multiseed"] + data["d8_nested_basis_continuation_multiseed"])
    factorization = all(d["dense_relative_factorization_error"] <= thresholds["factorization_error"] for d in data["operator_diagnostics"])
    k_axis = data["convergence"]["K_axis_D6"]
    k_monotone = all(b["energy"] <= a["energy"] + 1e-9 for a, b in zip(k_axis, k_axis[1:]))
    d_axis = data["convergence"]["D_axis_K4"]
    d_monotone = d_axis[1]["absolute_error_vs_D14"] <= d_axis[0]["absolute_error_vs_D14"] + 1e-9
    no_enumeration = all(p["structural_counts"]["enumerated_virtual_paths"] == 0 for p in all_points)
    accepted = stability and factorization and k_monotone and d_monotone and no_enumeration
    if accepted != data["acceptance"]["transferability_pass"]:
        raise AssertionError("recorded acceptance disagrees with recomputation")
    if not accepted:
        raise AssertionError("soft-Coulomb transferability criteria do not pass")
    return {"verified": True, "transferability_pass": True, "points": len(all_points)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", nargs="?", type=Path, default=Path("docs/experiments/results/phase28_soft_coulomb_transferability.json"))
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
