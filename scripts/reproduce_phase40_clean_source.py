"""Clean-source deterministic reproduction of the frozen Phase 40 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.benchmark_phase40_explicit_correlation_gate import run
    from scripts.verify_phase40_explicit_correlation_gate import verify
except ModuleNotFoundError:
    from benchmark_phase40_explicit_correlation_gate import run
    from verify_phase40_explicit_correlation_gate import verify


PRIMARY = Path("docs/experiments/results/phase40_explicit_correlation_gate.json")
REPRODUCTION = Path(
    "docs/experiments/results/phase42_phase40_clean_reproduction_full.json"
)
SUMMARY = Path("docs/experiments/results/phase42_phase40_clean_reproduction.json")
CHECKPOINT_DIR = Path("checkpoints/phase42_phase40_clean_reproduction")
TOLERANCE = 2e-10


def _sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _keyed(points: list[dict[str, Any]], labels: tuple[str, ...]):
    return {tuple(point[label] for label in labels): point for point in points}


def reproduce(
    primary_path: Path,
    reproduction_path: Path,
    summary_path: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    if reproduction_path.exists():
        raise FileExistsError(
            "clean reproduction output already exists; refusing an outcome-dependent overwrite"
        )
    if checkpoint_dir.exists():
        raise FileExistsError(
            "clean reproduction checkpoint directory already exists; refusing reuse"
        )
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    reproduced = run(reproduction_path, checkpoint_dir, resume=False)
    reconstruction = verify(reproduction_path)

    primary_correlated = _keyed(
        primary["correlated_points"], ("D", "P", "seed")
    )
    reproduced_correlated = _keyed(
        reproduced["correlated_points"], ("D", "P", "seed")
    )
    primary_noci = _keyed(primary["noci_points"], ("D", "K", "seed"))
    reproduced_noci = _keyed(reproduced["noci_points"], ("D", "K", "seed"))
    if primary_correlated.keys() != reproduced_correlated.keys():
        raise AssertionError("correlated clean-reproduction axes changed")
    if primary_noci.keys() != reproduced_noci.keys():
        raise AssertionError("NOCI clean-reproduction axes changed")

    correlated_fields = (
        "energy",
        "energy_variance",
        "raw_norm",
        "antisymmetry_residual",
        "correlator_symmetry_residual",
        "energy_uncertainty_q128_q160",
        "relative_norm_change_q128_q160",
    )
    noci_fields = (
        "energy",
        "energy_variance",
        "norm",
        "norm_error",
        "structural_antisymmetry_residual",
        "materialized_antisymmetry_residual",
        "polynomial_explicit_absolute_difference",
    )
    maximum_correlated_difference = 0.0
    maximum_noci_difference = 0.0
    for key in primary_correlated:
        for field in correlated_fields:
            maximum_correlated_difference = max(
                maximum_correlated_difference,
                abs(
                    primary_correlated[key][field]
                    - reproduced_correlated[key][field]
                ),
            )
        if primary_correlated[key]["validation"] != reproduced_correlated[key][
            "validation"
        ]:
            raise AssertionError(f"correlated validation changed at {key}")
    for key in primary_noci:
        for field in noci_fields:
            maximum_noci_difference = max(
                maximum_noci_difference,
                abs(primary_noci[key][field] - reproduced_noci[key][field]),
            )
        if primary_noci[key]["validation"] != reproduced_noci[key]["validation"]:
            raise AssertionError(f"NOCI validation changed at {key}")

    primary_pairs = primary["comparison"]["consecutive_advantage_pairs"]
    reproduced_pairs = reproduced["comparison"]["consecutive_advantage_pairs"]
    decisions_identical = (
        primary_pairs == reproduced_pairs
        and primary["acceptance"]["phase40_differentiator_pass"]
        == reproduced["acceptance"]["phase40_differentiator_pass"]
    )
    accepted = (
        reconstruction["verified"]
        and maximum_correlated_difference <= TOLERANCE
        and maximum_noci_difference <= TOLERANCE
        and decisions_identical
    )
    summary = {
        "schema_version": 1,
        "experiment": "phase42_clean_source_reproduction_of_phase40",
        "evidence_level": "clean-source deterministic numerical reproduction",
        "scientific_boundary": "same repository implementation and frozen seeds; not external scientific replication, many-particle scalability, or Paper B",
        "primary_artifact": str(primary_path),
        "reproduction_artifact": str(reproduction_path),
        "primary_sha256": _sha256(primary_path),
        "reproduction_sha256": _sha256(reproduction_path),
        "checkpoint_reuse": False,
        "correlated_points_compared": len(primary_correlated),
        "noci_points_compared": len(primary_noci),
        "maximum_correlated_observable_difference": maximum_correlated_difference,
        "maximum_noci_observable_difference": maximum_noci_difference,
        "tolerance": TOLERANCE,
        "primary_consecutive_advantage_pairs": primary_pairs,
        "reproduced_consecutive_advantage_pairs": reproduced_pairs,
        "decisions_identical": decisions_identical,
        "reconstruction": reconstruction,
        "clean_source_reproduction_pass": accepted,
        "publication_consequence": "no Paper B; external independent replication and many-particle controlled-contraction evidence remain required",
    }
    _write(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, default=PRIMARY)
    parser.add_argument("--reproduction", type=Path, default=REPRODUCTION)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    arguments = parser.parse_args()
    result = reproduce(
        arguments.primary,
        arguments.reproduction,
        arguments.summary,
        arguments.checkpoint_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
