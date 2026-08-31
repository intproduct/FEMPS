"""Build the frozen Phase 12 benchmark manifest and normalized seed records."""

from __future__ import annotations

import json
from pathlib import Path

from femps.benchmarks import (
    SoftCoulombBenchmarkPoint,
    direct_exterior_feasibility,
    soft_coulomb_point_from_training,
)


RESULTS = Path("docs/experiments/results")
LARGEST_N4_REFERENCE = 11.023082853675


def _direct_truth_maps() -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    paths = {
        4: RESULTS / "soft_coulomb_n4_truth_sweep.json",
        6: RESULTS / "soft_coulomb_n6_truth_sweep.json",
        8: RESULTS / "soft_coulomb_n8_truth_sweep.json",
    }
    by_basis = {}
    largest = {}
    for particles, path in paths.items():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        points = artifact["basis_scan"]
        for point in points:
            by_basis[(particles, point["D"])] = point["ground_energy"]
        largest[particles] = points[-1]["ground_energy"]
    return by_basis, largest


def _separate_error_axes(
    point: dict, direct_by_basis: dict, largest_by_particles: dict
) -> None:
    key = (point["particles"], point["basis_order"])
    if key not in direct_by_basis:
        return
    direct = direct_by_basis[key]
    largest = largest_by_particles[point["particles"]]
    finite = point["finite_basis_reference_energy"]
    point["direct_dense_same_basis_reference_energy"] = direct
    point["largest_basis_reference_energy"] = largest
    point["operator_error_estimate"] = finite - direct
    point["basis_error_estimate"] = direct - largest
    point["total_error_estimate"] = point["energy"] - largest


def main() -> None:
    direct_by_basis, largest_by_particles = _direct_truth_maps()
    k4 = json.loads(
        (RESULTS / "soft_coulomb_conditioned_d10_k4_seeds.json").read_text(
            encoding="utf-8"
        )
    )
    k5 = json.loads(
        (RESULTS / "soft_coulomb_conditioned_d10_k5_seeds.json").read_text(
            encoding="utf-8"
        )
    )
    points = []
    for artifact, label in ((k4, "k4"), (k5, "k5")):
        for run in artifact["runs"]:
            training = run["training"] if label == "k4" else run["joint"]
            point = soft_coulomb_point_from_training(
                f"n4-d10-{label}-s{run['seed']}",
                training,
                run["conditioning"],
                largest_basis_reference_energy=LARGEST_N4_REFERENCE,
            )
            points.append(point.to_dict())

    d12_path = RESULTS / "soft_coulomb_conditioned_d12_k5_seeds.json"
    if d12_path.exists():
        d12 = json.loads(d12_path.read_text(encoding="utf-8"))
        points.extend(run["normalized_point"] for run in d12["runs"])

    point_ids = {point["point_id"] for point in points}
    for path in sorted(RESULTS.glob("soft_coulomb_*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        point = artifact.get("normalized_point")
        if point is not None and point["point_id"] not in point_ids:
            points.append(point)
            point_ids.add(point["point_id"])
    for point in points:
        _separate_error_axes(point, direct_by_basis, largest_by_particles)
        SoftCoulombBenchmarkPoint(**point).validate()

    requested_grid = [
        {"N": 4, "D": d, "K": list(range(1, 7)), "Q": 128}
        for d in (8, 10, 12, 14)
    ] + [
        {"N": 6, "D": d, "K": [1, 2, 3], "Q": 128}
        for d in (8, 10, 12)
    ] + [
        {"N": 8, "D": d, "K": [1, 2, 3], "Q": 128}
        for d in (10, 12)
    ]
    for entry in requested_grid:
        entry["direct_exterior_truth"] = direct_exterior_feasibility(
            entry["N"], entry["D"]
        ).to_dict()

    manifest = {
        "schema_version": 1,
        "experiment": "phase12_controlled_soft_coulomb_benchmark_matrix",
        "evidence_level": "numerical",
        "method_scope": "fixed_number_finite_AGP_subclass_not_generic_FEMPS",
        "dense_truth_maximum_exterior_dimension": 1200,
        "primary_seeds": [301, 302, 303],
        "required_fields": list(points[0]),
        "requested_grid": requested_grid,
        "completed_normalized_points": points,
    }
    output = RESULTS / "phase12_benchmark_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(points)} normalized points and "
        f"{len(requested_grid)} grid blocks"
    )


if __name__ == "__main__":
    main()
