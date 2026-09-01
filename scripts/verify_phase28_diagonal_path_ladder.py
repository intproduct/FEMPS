"""Verify the admitted E1/E2/E3 subset of the Phase 28 ladder artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_artifact(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["evidence_level"] != "numerical":
        raise AssertionError("diagonal-path ladder must remain numerical evidence")
    if "first-quantized continuous functional-basis" not in data["state_definition"]:
        raise AssertionError("ladder state definition changed quantization or basis")
    admitted = [
        point
        for point in data["points"]
        if point["gate_label"] in {"E1", "E2_D", "E2_K", "E2_DK", "E3"}
    ]
    if len(admitted) != 7:
        raise AssertionError("expected the registered E1/E2/E3 points")
    for point in admitted:
        if not (
            point["completed"]
            and point["method"] == "diagonal_path_femps"
            and point["norm_error"] <= 1e-10
            and point["structural_antisymmetry_residual"] <= 1e-12
            and point["materialized_antisymmetry_residual"] <= 1e-12
            and point["polynomial_explicit_absolute_difference"] <= 1e-10
            and point["structural_counts"]["enumerated_virtual_paths"] == 0
            and point["structural_counts"]["materialized_particle_coefficients"]
            == 0
        ):
            raise AssertionError(f"ladder invariant failed: {point['point_id']}")
    e1 = next(point for point in admitted if point["gate_label"] == "E1")
    e3 = next(point for point in admitted if point["gate_label"] == "E3")
    if abs(e1["energy"] - 2.0) > 1e-12 or e1["energy_variance"] > 1e-12:
        raise AssertionError("E1 noninteracting pair is not exact")
    if abs(e3["energy"] - 8.0) > 1e-12 or e3["energy_variance"] > 1e-12:
        raise AssertionError("E3 noninteracting N4 Slater is not exact")
    k_axis = sorted(
        (
            point
            for point in admitted
            if point["config"]["particles"] == 2
            and point["config"]["basis_order"] == 6
            and point["config"]["kappa"] == 0.35
        ),
        key=lambda point: point["config"]["terms"],
    )
    if [point["config"]["terms"] for point in k_axis] != [1, 2, 4]:
        raise AssertionError("E2 K axis must be K=1,2,4")
    if any(
        right["energy"] > left["energy"] + 1e-9
        for left, right in zip(k_axis, k_axis[1:])
    ):
        raise AssertionError("E2 K-axis energy is not nonincreasing")
    d_axis = sorted(
        (
            point
            for point in admitted
            if point["config"]["particles"] == 2
            and point["config"]["terms"] == 4
            and point["config"]["kappa"] == 0.35
        ),
        key=lambda point: point["config"]["basis_order"],
    )
    if [point["config"]["basis_order"] for point in d_axis] != [4, 6, 8]:
        raise AssertionError("E2 D axis must be D=4,6,8")
    if any(
        abs(right["error_vs_continuum"]) > abs(left["error_vs_continuum"]) + 1e-9
        for left, right in zip(d_axis, d_axis[1:])
    ):
        raise AssertionError("E2 D-axis continuum error is not nonincreasing")
    return {
        "verified": True,
        "admitted_points": len(admitted),
        "E2_K4_D8_continuum_error": d_axis[-1]["error_vs_continuum"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path("docs/experiments/results/phase28_diagonal_path_ladder.json"),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
