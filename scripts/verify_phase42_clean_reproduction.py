"""Verify the persisted clean-source Phase 40 reproduction comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.verify_phase40_explicit_correlation_gate import verify as verify_phase40
except ModuleNotFoundError:
    from verify_phase40_explicit_correlation_gate import verify as verify_phase40


DEFAULT_SUMMARY = Path(
    "docs/experiments/results/phase42_phase40_clean_reproduction.json"
)


def _sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _keyed(points: list[dict[str, Any]], labels: tuple[str, ...]):
    return {tuple(point[label] for label in labels): point for point in points}


def verify(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != 1:
        raise ValueError("unsupported clean-reproduction schema")
    if summary.get("evidence_level") != "clean-source deterministic numerical reproduction":
        raise ValueError("clean-reproduction evidence boundary changed")
    if "not external scientific replication" not in summary["scientific_boundary"]:
        raise ValueError("external-replication boundary is missing")
    primary_path = Path(summary["primary_artifact"])
    reproduction_path = Path(summary["reproduction_artifact"])
    if _sha256(primary_path) != summary["primary_sha256"]:
        raise AssertionError("primary artifact hash mismatch")
    if _sha256(reproduction_path) != summary["reproduction_sha256"]:
        raise AssertionError("reproduction artifact hash mismatch")
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
    if primary["frozen_config"] != reproduction["frozen_config"]:
        raise AssertionError("clean reproduction changed the frozen config")
    if primary["reference"] != reproduction["reference"]:
        raise AssertionError("clean reproduction changed the energy reference")

    maximum_correlated_difference = 0.0
    primary_correlated = _keyed(primary["correlated_points"], ("D", "P", "seed"))
    reproduced_correlated = _keyed(
        reproduction["correlated_points"], ("D", "P", "seed")
    )
    if primary_correlated.keys() != reproduced_correlated.keys():
        raise AssertionError("clean correlated axes mismatch")
    for key in primary_correlated:
        for field in (
            "energy",
            "energy_variance",
            "raw_norm",
            "antisymmetry_residual",
            "correlator_symmetry_residual",
            "energy_uncertainty_q128_q160",
            "relative_norm_change_q128_q160",
        ):
            maximum_correlated_difference = max(
                maximum_correlated_difference,
                abs(primary_correlated[key][field] - reproduced_correlated[key][field]),
            )

    maximum_noci_difference = 0.0
    primary_noci = _keyed(primary["noci_points"], ("D", "K", "seed"))
    reproduced_noci = _keyed(reproduction["noci_points"], ("D", "K", "seed"))
    if primary_noci.keys() != reproduced_noci.keys():
        raise AssertionError("clean NOCI axes mismatch")
    for key in primary_noci:
        for field in (
            "energy",
            "energy_variance",
            "norm",
            "norm_error",
            "structural_antisymmetry_residual",
            "materialized_antisymmetry_residual",
            "polynomial_explicit_absolute_difference",
        ):
            maximum_noci_difference = max(
                maximum_noci_difference,
                abs(primary_noci[key][field] - reproduced_noci[key][field]),
            )

    primary_verification = verify_phase40(primary_path)
    reproduction_verification = verify_phase40(reproduction_path)
    pairs_equal = (
        primary_verification["consecutive_advantage_pairs"]
        == reproduction_verification["consecutive_advantage_pairs"]
        == summary["primary_consecutive_advantage_pairs"]
        == summary["reproduced_consecutive_advantage_pairs"]
    )
    accepted = (
        primary_verification["verified"]
        and reproduction_verification["verified"]
        and maximum_correlated_difference <= summary["tolerance"]
        and maximum_noci_difference <= summary["tolerance"]
        and pairs_equal
        and summary["decisions_identical"]
        and not summary["checkpoint_reuse"]
    )
    if accepted != summary["clean_source_reproduction_pass"]:
        raise AssertionError("clean-source reproduction decision mismatch")
    return {
        "verified": True,
        "clean_source_reproduction_pass": accepted,
        "maximum_correlated_observable_difference": maximum_correlated_difference,
        "maximum_noci_observable_difference": maximum_noci_difference,
        "consecutive_advantage_pairs": primary_verification[
            "consecutive_advantage_pairs"
        ],
        "external_independent_replication_complete": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_SUMMARY)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.input), indent=2))


if __name__ == "__main__":
    main()
