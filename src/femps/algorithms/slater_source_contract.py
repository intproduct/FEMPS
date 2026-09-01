"""Versioned contracts for the clean Slater-source FEMPS command."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


SLATER_SOURCE_CHECKPOINT_SCHEMA_VERSION = 1
SLATER_SOURCE_RESULT_SCHEMA_VERSION = 1
SLATER_SOURCE_METHOD = "slater_source_adaptive_diagonal_path_femps"


def validate_slater_source_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_initial_source_identity: Mapping[str, Any] | None = None,
    expected_operator_identity: Mapping[str, Any] | None = None,
) -> None:
    """Validate a command-level checkpoint and optional resume identities."""

    required = {
        "schema_version",
        "method",
        "evidence_level",
        "scientific_boundary",
        "config",
        "config_sha256",
        "initial_source_identity",
        "operator_identity",
        "source_completed",
        "source_result",
        "accepted_source_raw",
        "adaptive_result",
        "current_terms",
        "completed",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Slater-source checkpoint is missing fields: {sorted(missing)}")
    if payload["schema_version"] != SLATER_SOURCE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported Slater-source checkpoint schema version")
    if payload["method"] != SLATER_SOURCE_METHOD:
        raise ValueError("unexpected Slater-source checkpoint method")
    if payload["evidence_level"] != "numerical":
        raise ValueError("Slater-source checkpoint must be numerical evidence")
    for expected, observed, label in (
        (expected_config, payload["config"], "configuration"),
        (
            expected_initial_source_identity,
            payload["initial_source_identity"],
            "initial source identity",
        ),
        (expected_operator_identity, payload["operator_identity"], "operator identity"),
    ):
        if expected is not None and dict(observed) != dict(expected):
            raise ValueError(f"Slater-source checkpoint {label} does not match")
    accepted = payload["accepted_source_raw"]
    if payload["source_completed"]:
        if not isinstance(accepted, torch.Tensor) or accepted.ndim != 3:
            raise ValueError("completed source requires accepted (K,D,N) orbitals")
        if accepted.shape[0] != 1:
            raise ValueError("accepted clean source must contain exactly one Slater term")
        if payload["source_result"] is None or not payload["source_result"]["completed"]:
            raise ValueError("source completion disagrees with its result")
    elif accepted is not None:
        raise ValueError("incomplete source cannot expose accepted source orbitals")
    if payload["completed"]:
        adaptive = payload["adaptive_result"]
        if adaptive is None or not adaptive["completed"]:
            raise ValueError("completed command requires a completed adaptive result")
        if payload["current_terms"] != payload["config"]["max_terms"]:
            raise ValueError("completed command did not reach its external maximum K")


def load_slater_source_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_config: Mapping[str, Any] | None = None,
    expected_initial_source_identity: Mapping[str, Any] | None = None,
    expected_operator_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate a versioned command-level checkpoint."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Slater-source checkpoint payload must be a mapping")
    validate_slater_source_checkpoint(
        payload,
        expected_config=expected_config,
        expected_initial_source_identity=expected_initial_source_identity,
        expected_operator_identity=expected_operator_identity,
    )
    return payload


def validate_slater_source_result(
    result: Mapping[str, Any], *, require_completed: bool = False
) -> None:
    """Validate scientific fields and no-enumeration boundaries of a result."""

    required = {
        "schema_version",
        "method",
        "evidence_level",
        "scientific_boundary",
        "config",
        "config_sha256",
        "source_construction",
        "initial_source_identity",
        "accepted_source_identity",
        "operator_identity",
        "operator_metadata",
        "source_result",
        "adaptive_result",
        "stages",
        "current_terms",
        "completed",
        "resumed",
        "total_elapsed_seconds_this_call",
        "cpu_memory",
        "peak_cpu_rss_bytes",
        "automatic_stopping_rule",
        "external_max_terms_required",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Slater-source result is missing fields: {sorted(missing)}")
    if result["schema_version"] != SLATER_SOURCE_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported Slater-source result schema version")
    if result["method"] != SLATER_SOURCE_METHOD or result["evidence_level"] != "numerical":
        raise ValueError("unexpected Slater-source result method/evidence label")
    construction = result["source_construction"]
    if construction.get("kind") != "canonical_lowest_functional_basis_slater":
        raise ValueError("source is not the canonical lowest-orbital Slater")
    if construction.get("historical_checkpoint_used") or construction.get(
        "ci_initializer_used"
    ):
        raise ValueError("clean source cannot use historical or CI initialization")
    stages = result["stages"]
    if not stages or stages[0].get("terms") != 1:
        raise ValueError("result must start with the optimized K1 source")
    expected_terms = list(range(1, int(result["current_terms"]) + 1))
    if [int(stage["terms"]) for stage in stages] != expected_terms:
        raise ValueError("result stages must be consecutive from K1")
    for stage in stages:
        solver = stage["optimizer_result"]
        counts = solver["structural_counts"]
        if solver["structural_antisymmetry_residual"] < 0:
            raise ValueError("antisymmetry residual must be nonnegative")
        if counts["enumerated_virtual_paths"] != 0:
            raise ValueError("production enumerated virtual paths")
        if counts["materialized_particle_coefficients"] != 0:
            raise ValueError("production materialized D^N coefficients")
        if "norm_error" not in solver or "energy" not in solver:
            raise ValueError("stage lacks norm or energy diagnostics")
    if result["automatic_stopping_rule"] != "not_admitted":
        raise ValueError("automatic stopping is outside the admitted command")
    if not result["external_max_terms_required"]:
        raise ValueError("command must require an external maximum K")
    if require_completed and not result["completed"]:
        raise ValueError("Slater-source command is incomplete")
    if result["completed"] and result["current_terms"] != result["config"]["max_terms"]:
        raise ValueError("completed result did not reach the external maximum K")


__all__ = [
    "SLATER_SOURCE_CHECKPOINT_SCHEMA_VERSION",
    "SLATER_SOURCE_METHOD",
    "SLATER_SOURCE_RESULT_SCHEMA_VERSION",
    "load_slater_source_checkpoint",
    "validate_slater_source_checkpoint",
    "validate_slater_source_result",
]
