"""Independently verify the Phase 32 N=6 resource-preflight artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_DIMENSIONS = (8, 10, 12)
EXPECTED_RANKS = {8: 15, 10: 19, 12: 23}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["schema_version"] != 1:
        raise AssertionError("unexpected schema")
    if artifact["evidence_level"] != "numerical":
        raise AssertionError("resource preflight must remain numerical evidence")
    if "not an asymptotic fit" not in artifact["scientific_boundary"]:
        raise AssertionError("scientific boundary was weakened")

    contract = artifact["production_contract"]
    if contract["device"] != "cpu":
        raise AssertionError("registered production device changed")
    if contract["adam_steps"] != 160 or contract["lbfgs_steps"] != 80:
        raise AssertionError("registered optimization budget changed")
    if any(
        (
            contract["truth_state_initialization"],
            contract["enumerate_virtual_paths"],
            contract["materialize_particle_tensor"],
        )
    ):
        raise AssertionError("production structural boundary changed")
    backend = artifact["backend_audit"]
    if "Blackwell" not in backend["blackwell_device_name"]:
        raise AssertionError("Blackwell diagnostic device changed")
    if not backend["complex128_autograd_probe_pass"]:
        raise AssertionError("Blackwell complex128 probe failed")
    if backend["N6_D10_K4_registered_runtime_gate"] != {
        "status": "stopped_after_600_second_limit",
        "completed_scientific_point": False,
        "diagnosis": "Python-level transition/factor loops launch many small kernels",
    }:
        raise AssertionError("Blackwell runtime-gate record changed")

    audit = artifact["operator_audit"]
    if tuple(point["D"] for point in audit) != EXPECTED_DIMENSIONS:
        raise AssertionError("basis audit axis changed")
    for point in audit:
        dimension = point["D"]
        rank = EXPECTED_RANKS[dimension]
        if point["physical_operator_svd_rank"] != rank:
            raise AssertionError("operator-SVD rank changed")
        if point["exterior_ci_dimension"] != math.comb(dimension, 6):
            raise AssertionError("CI dimension is inconsistent")
        if point["forbidden_particle_tensor_coefficients"] != dimension**6:
            raise AssertionError("particle tensor count is inconsistent")
        counts = point["structural_counts"]
        if counts["transition_pairs"] != 16:
            raise AssertionError("transition-pair count is inconsistent")
        if counts["enumerated_virtual_paths"] != 0:
            raise AssertionError("virtual paths may not be enumerated")
        if counts["materialized_particle_coefficients"] != 0:
            raise AssertionError("production particle tensor was materialized")

    anchor = artifact["empirical_anchor"]
    anchor_path = Path(anchor["artifact"])
    if _sha256(anchor_path) != anchor["sha256"]:
        raise AssertionError("empirical anchor hash changed")
    source = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor_time = max(
        point["total_elapsed_seconds_this_call"] for point in source["points"]
    )
    anchor_rss = max(point["peak_cpu_rss_bytes"] for point in source["points"])
    estimate = artifact["D12_estimate"]
    ratio = (12 / 10) ** 2 * (EXPECTED_RANKS[12] / EXPECTED_RANKS[10])
    if not math.isclose(estimate["work_ratio_vs_D10"], ratio, rel_tol=1e-15):
        raise AssertionError("work ratio is inconsistent")
    expected_time = anchor_time * ratio * estimate["multiplicative_margin"]
    expected_rss = math.ceil(
        anchor_rss * ratio * estimate["multiplicative_margin"]
    )
    if not math.isclose(
        estimate["estimated_wall_time_seconds"], expected_time, rel_tol=1e-15
    ):
        raise AssertionError("wall-time estimate is inconsistent")
    if estimate["estimated_peak_cpu_rss_bytes"] != expected_rss:
        raise AssertionError("RSS estimate is inconsistent")
    admitted = bool(
        expected_time <= estimate["registered_wall_time_cap_seconds"]
        and expected_rss <= estimate["registered_peak_cpu_rss_cap_bytes"]
    )
    if estimate["admitted_before_production"] != admitted or not admitted:
        raise AssertionError("D=12 preflight admission is inconsistent")
    if not all(artifact["acceptance"].values()):
        raise AssertionError("resource preflight did not pass")
    return {
        "verified": True,
        "dimensions": list(EXPECTED_DIMENSIONS),
        "D12_estimated_wall_time_seconds": expected_time,
        "D12_estimated_peak_cpu_rss_bytes": expected_rss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        type=Path,
        nargs="?",
        default=Path("docs/experiments/results/phase32_n6_resource_audit.json"),
    )
    print(json.dumps(verify_artifact(parser.parse_args().artifact), indent=2))


if __name__ == "__main__":
    main()
