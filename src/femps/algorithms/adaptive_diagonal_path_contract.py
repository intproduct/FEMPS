"""Versioned records for bounded adaptive diagonal-path FEMPS runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


ADAPTIVE_DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION = 1
ADAPTIVE_DIAGONAL_PATH_RESULT_SCHEMA_VERSION = 1

_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "evidence_level",
        "scientific_boundary",
        "source_identity",
        "operator_identity",
        "adaptive_config",
        "optimizer_template",
        "start_terms",
        "current_terms",
        "current_orbitals",
        "stages",
        "completed",
    }
)

_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "evidence_level",
        "scientific_boundary",
        "source_identity",
        "operator_identity",
        "adaptive_config",
        "optimizer_template",
        "start_terms",
        "current_terms",
        "completed",
        "resumed",
        "stages_completed_this_call",
        "stages",
        "final_energy",
        "structural_antisymmetry_residual",
        "enumerated_virtual_paths",
        "materialized_particle_coefficients",
        "automatic_stopping_rule",
        "external_max_terms_required",
    }
)


def _validate_shared(payload: Mapping[str, Any], required: frozenset[str]) -> None:
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"adaptive diagonal-path record is missing: {sorted(missing)}")
    if payload["method"] != "bounded_adaptive_diagonal_path_femps":
        raise ValueError("adaptive record has the wrong method")
    if payload["evidence_level"] != "numerical":
        raise ValueError("adaptive solver records must be labeled numerical")
    if (
        "external_max_terms_required" in payload
        and not payload["external_max_terms_required"]
    ):
        raise ValueError("adaptive solver must retain an external maximum K")
    if payload.get("automatic_stopping_rule") not in (None, "not_admitted"):
        raise ValueError("automatic stopping is not admitted")
    if payload["current_terms"] < payload["start_terms"]:
        raise ValueError("adaptive current K cannot be below source K")
    max_terms = payload["adaptive_config"]["max_terms"]
    if payload["current_terms"] > max_terms:
        raise ValueError("adaptive current K exceeds the external maximum")
    if payload["completed"] != (payload["current_terms"] == max_terms):
        raise ValueError("adaptive completion flag disagrees with current K")


def validate_adaptive_diagonal_path_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_source_identity: Mapping[str, Any] | None = None,
    expected_operator_identity: Mapping[str, Any] | None = None,
    expected_adaptive_config: Mapping[str, Any] | None = None,
    expected_optimizer_template: Mapping[str, Any] | None = None,
) -> None:
    """Validate an outer stage-level adaptive checkpoint and its identities."""

    _validate_shared(payload, _CHECKPOINT_FIELDS)
    if (
        payload["schema_version"]
        != ADAPTIVE_DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported adaptive checkpoint schema version")
    for observed, expected, label in (
        (payload["source_identity"], expected_source_identity, "source"),
        (payload["operator_identity"], expected_operator_identity, "operator"),
        (payload["adaptive_config"], expected_adaptive_config, "adaptive config"),
        (payload["optimizer_template"], expected_optimizer_template, "optimizer template"),
    ):
        if expected is not None and observed != dict(expected):
            raise ValueError(f"adaptive checkpoint {label} does not match")
    orbitals = payload["current_orbitals"]
    if not isinstance(orbitals, torch.Tensor) or orbitals.ndim != 3:
        raise ValueError("adaptive checkpoint orbitals must have shape (K,D,N)")
    expected_shape = (
        payload["current_terms"],
        payload["optimizer_template"]["basis_order"],
        payload["optimizer_template"]["particles"],
    )
    if tuple(orbitals.shape) != expected_shape:
        raise ValueError("adaptive checkpoint orbitals disagree with current K")
    if len(payload["stages"]) != payload["current_terms"] - payload["start_terms"]:
        raise ValueError("adaptive checkpoint stage count disagrees with current K")


def load_adaptive_diagonal_path_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_source_identity: Mapping[str, Any] | None = None,
    expected_operator_identity: Mapping[str, Any] | None = None,
    expected_adaptive_config: Mapping[str, Any] | None = None,
    expected_optimizer_template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate an adaptive stage checkpoint."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("adaptive checkpoint payload must be a mapping")
    validate_adaptive_diagonal_path_checkpoint(
        payload,
        expected_source_identity=expected_source_identity,
        expected_operator_identity=expected_operator_identity,
        expected_adaptive_config=expected_adaptive_config,
        expected_optimizer_template=expected_optimizer_template,
    )
    return dict(payload)


def validate_adaptive_diagonal_path_result(
    record: Mapping[str, Any], *, require_completed: bool = False
) -> None:
    """Validate the public adaptive-run result contract."""

    _validate_shared(record, _RESULT_FIELDS)
    if record["schema_version"] != ADAPTIVE_DIAGONAL_PATH_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported adaptive result schema version")
    if require_completed and not record["completed"]:
        raise ValueError("adaptive result is incomplete")
    if record["structural_antisymmetry_residual"] is None:
        raise ValueError("adaptive result must report antisymmetry residual")
    if record["enumerated_virtual_paths"] != 0:
        raise ValueError("adaptive production may not enumerate virtual paths")
    if record["materialized_particle_coefficients"] != 0:
        raise ValueError("adaptive production may not materialize a D^N tensor")
    if len(record["stages"]) != record["current_terms"] - record["start_terms"]:
        raise ValueError("adaptive result stage count disagrees with current K")
    for stage in record["stages"]:
        required_stage = {
            "target_terms",
            "candidate_seed",
            "optimizer_seed",
            "selected_candidate",
            "predicted_improvement",
            "source_nesting_max_abs_error",
            "optimizer_result",
            "optimized_orbitals_sha256",
        }
        if missing := required_stage.difference(stage):
            raise ValueError(f"adaptive stage is missing: {sorted(missing)}")
        result = stage["optimizer_result"]
        if result["structural_antisymmetry_residual"] is None:
            raise ValueError("every adaptive stage must report antisymmetry residual")
        counts = result["structural_counts"]
        if counts["enumerated_virtual_paths"] != 0:
            raise ValueError("adaptive stage enumerated virtual paths")
        if counts["materialized_particle_coefficients"] != 0:
            raise ValueError("adaptive stage materialized a D^N tensor")


__all__ = [
    "ADAPTIVE_DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION",
    "ADAPTIVE_DIAGONAL_PATH_RESULT_SCHEMA_VERSION",
    "load_adaptive_diagonal_path_checkpoint",
    "validate_adaptive_diagonal_path_checkpoint",
    "validate_adaptive_diagonal_path_result",
]
