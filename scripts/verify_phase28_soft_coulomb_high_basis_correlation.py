"""Independently verify the recorded D12,K4 -> K5 correlation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_artifact(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["evidence_level"] != "numerical":
        raise AssertionError("correlation audit must be labeled numerical evidence")
    source = data["source_K4_point"]
    point = data["K5_point"]
    thresholds = data["thresholds"]
    if [entry["K"] for entry in data["correlation_axis"]] != [4, 5]:
        raise AssertionError("expected the registered K=4,5 axis")
    source_nested = (
        data["diagnostics"]["initial_nested_energy"] <= source["energy"] + 1e-9
    )
    optimized_nonworsening = (
        point["energy"] <= data["diagnostics"]["initial_nested_energy"] + 1e-9
    )
    dense_error = point["energy"] - data["dense_ci_audit"]["energy"]
    state_pass = bool(
        point["completed"]
        and -1e-9 <= dense_error <= thresholds["dense_ci_error"]
        and point["energy_variance"] <= thresholds["variance"]
        and point["norm_error"] <= thresholds["norm_error"]
        and point["structural_antisymmetry_residual"]
        <= thresholds["antisymmetry_residual"]
        and point["materialized_antisymmetry_residual"]
        <= thresholds["antisymmetry_residual"]
        and point["structural_counts"]["enumerated_virtual_paths"] == 0
        and point["peak_cpu_rss_bytes"] > 0
    )
    operator_pass = bool(
        data["operator_audit"]["backend"] == "physical_operator_svd"
        and data["operator_audit"]["dense_relative_factorization_error"]
        <= thresholds["factorization_error"]
        and abs(
            point["finite_basis_reference_energy"]
            - data["dense_ci_audit"]["energy"]
        )
        <= thresholds["factorization_error"]
    )
    audit_pass = source_nested and optimized_nonworsening and state_pass and operator_pass
    reduction = (
        source["error_vs_dense_quadrature_ci"] - dense_error
    ) / source["error_vs_dense_quadrature_ci"]
    material = bool(
        reduction >= thresholds["material_error_reduction_fraction"]
        and point["energy_variance"] <= source["energy_variance"]
    )
    recorded = data["acceptance"]
    if (
        state_pass != recorded["state_pass"]
        or operator_pass != recorded["operator_pass"]
        or audit_pass != recorded["audit_pass"]
        or material != recorded["material_improvement"]
    ):
        raise AssertionError("recorded correlation acceptance disagrees")
    if not audit_pass:
        raise AssertionError("high-basis correlation audit does not pass")
    return {
        "verified": True,
        "audit_pass": True,
        "material_improvement": material,
        "error_reduction_fraction": reduction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path(
            "docs/experiments/results/"
            "phase28_soft_coulomb_high_basis_correlation.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
