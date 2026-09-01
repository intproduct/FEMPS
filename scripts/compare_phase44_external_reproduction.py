"""Compare a clean Phase 44 rerun with the authenticated primary result.

This script verifies the reproduction from its own checkpoints and raw
coordinate archives, then compares the frozen design and scientific decisions
with the committed primary artifact.  Passing this numerical comparison is
necessary but not sufficient for *external independent replication*: named
human provenance and conflict disclosure remain outside what code can attest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from scripts.verify_phase44_n4_explicit_correlation_d_gate import verify
except ModuleNotFoundError:
    from verify_phase44_n4_explicit_correlation_d_gate import verify


DEFAULT_PRIMARY = Path(
    "docs/experiments/results/phase44_n4_explicit_correlation_d_gate.json"
)
DEFAULT_PRIMARY_MANIFEST = Path(
    "docs/experiments/results/phase44_optimizer_checkpoint_manifest.json"
)
IDENTITY_FIELDS = (
    "schema_version",
    "experiment",
    "evidence_level",
    "scientific_boundary",
    "adr",
    "frozen_axes",
    "frozen_exponents",
    "optimizer_configs",
    "comparators",
    "source_hashes",
)


def _normalized_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _decision_signature(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "acceptance": artifact["acceptance"],
        "selected_lineages": [
            [record["D"], record["selected_lineage"]]
            for record in artifact["selection_choices"]
        ],
        "selection_gate_maps": [
            [record["D"], record["lineage"], record["gates"]]
            for record in artifact["selection_runs"]
        ],
        "confirmation_gate_maps": [
            [record["D"], record["confirmation_index"], record["gates"]]
            for record in artifact["confirmation_runs"]
        ],
        "D_monotonicity_passes": [
            [record["lower_D"], record["upper_D"], record["pass"]]
            for record in artifact["D_monotonicity"]
        ],
        "point_advantage_passes": [
            [record["D"], record["pass"]]
            for record in artifact["point_advantage_gates"]
        ],
        "consecutive_advantage_pairs": artifact["consecutive_advantage_pairs"],
    }


def _maximum_antisymmetry_residual(artifact: dict[str, Any]) -> float:
    residuals = []
    for record in artifact["optimizer_runs"]:
        residuals.extend(
            step["antisymmetry_residual"] for step in record["history"]
        )
    for group in ("selection_runs", "confirmation_runs"):
        residuals.extend(
            record["symmetry"]["antisymmetry_residual"]
            for record in artifact[group]
        )
    return max(residuals)


def _forbidden_materialization_absent(artifact: dict[str, Any]) -> bool:
    records = (
        artifact["optimizer_runs"]
        + artifact["selection_runs"]
        + artifact["confirmation_runs"]
    )
    return all(
        record["materialization"].get("D_to_the_N_tensor") is False
        and record["materialization"].get(
            "full_alternating_coefficient_tensor"
        )
        is False
        and record["materialization"].get("virtual_paths") == 0
        for record in records
    )


def compare(
    reproduction_path: Path,
    primary_path: Path = DEFAULT_PRIMARY,
    primary_manifest_path: Path = DEFAULT_PRIMARY_MANIFEST,
) -> dict[str, Any]:
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
    manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    primary_authenticated = _normalized_sha256(primary_path) == manifest[
        "phase44_artifact_normalized_sha256"
    ]
    if not primary_authenticated:
        raise AssertionError("primary Phase 44 artifact does not match its manifest")

    identity_matches = {
        field: reproduction.get(field) == primary.get(field)
        for field in IDENTITY_FIELDS
    }
    if not all(identity_matches.values()):
        return {
            "schema_version": 1,
            "primary_authenticated": True,
            "reproduction_self_verified": False,
            "identity_matches": identity_matches,
            "numerical_reproduction_pass": False,
            "external_independent_replication_complete": False,
            "external_independence_requires_named_human_attestation": True,
            "paper_b_authorized": False,
        }

    try:
        verification = verify(reproduction_path, None)
    except Exception as error:  # report a failed artifact without claiming review
        return {
            "schema_version": 1,
            "primary_authenticated": True,
            "reproduction_self_verified": False,
            "identity_matches": identity_matches,
            "verification_error": f"{type(error).__name__}: {error}",
            "numerical_reproduction_pass": False,
            "external_independent_replication_complete": False,
            "external_independence_requires_named_human_attestation": True,
            "paper_b_authorized": False,
        }

    energy_comparisons = []
    for basis_order in primary["frozen_axes"]["D"]:
        reference = primary["combined_confirmations"][str(basis_order)]
        rerun = reproduction["combined_confirmations"][str(basis_order)]
        difference = abs(
            rerun["inverse_variance_energy"]
            - reference["inverse_variance_energy"]
        )
        combined_standard_error = math.sqrt(
            rerun["inverse_variance_standard_error"] ** 2
            + reference["inverse_variance_standard_error"] ** 2
        )
        allowance = 5.0 * combined_standard_error + 2e-4
        energy_comparisons.append(
            {
                "D": basis_order,
                "primary_energy": reference["inverse_variance_energy"],
                "reproduction_energy": rerun["inverse_variance_energy"],
                "absolute_difference": difference,
                "combined_standard_error": combined_standard_error,
                "statistical_z": difference / combined_standard_error,
                "allowance": allowance,
                "pass": difference <= allowance,
            }
        )

    decisions_identical = _decision_signature(reproduction) == _decision_signature(
        primary
    )
    maximum_residual = _maximum_antisymmetry_residual(reproduction)
    materialization_pass = _forbidden_materialization_absent(reproduction)
    expected_failed_aggregate = (
        reproduction["acceptance"]["phase44_interacting_d_gate_pass"] is False
    )
    expected_low_D_subgate = (
        reproduction["acceptance"]["two_consecutive_D_advantage_pass"] is True
        and reproduction["consecutive_advantage_pairs"] == [[4, 6]]
    )
    numerical_pass = all(
        (
            verification["verified"],
            verification["checkpoint_verification_mode"]
            == "artifact_self_contained",
            decisions_identical,
            all(record["pass"] for record in energy_comparisons),
            maximum_residual <= 1e-12,
            materialization_pass,
            expected_failed_aggregate,
            expected_low_D_subgate,
        )
    )
    return {
        "schema_version": 1,
        "primary_authenticated": True,
        "reproduction_self_verified": verification["verified"],
        "checkpoint_verification_mode": verification[
            "checkpoint_verification_mode"
        ],
        "identity_matches": identity_matches,
        "decisions_identical": decisions_identical,
        "energy_comparisons": energy_comparisons,
        "maximum_antisymmetry_residual": maximum_residual,
        "antisymmetry_tolerance": 1e-12,
        "forbidden_materialization_absent": materialization_pass,
        "failed_aggregate_reproduced": expected_failed_aggregate,
        "low_D_4_6_subgate_reproduced": expected_low_D_subgate,
        "numerical_reproduction_pass": numerical_pass,
        "external_independent_replication_complete": False,
        "external_independence_requires_named_human_attestation": True,
        "paper_b_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument(
        "--primary-manifest", type=Path, default=DEFAULT_PRIMARY_MANIFEST
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = compare(
        arguments.reproduction, arguments.primary, arguments.primary_manifest
    )
    rendered = json.dumps(result, indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["numerical_reproduction_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
