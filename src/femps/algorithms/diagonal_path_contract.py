"""Versioned public records for the restricted diagonal-path FEMPS solver."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION = 1
DIAGONAL_PATH_RESULT_SCHEMA_VERSION = 2

_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "config",
        "operator_id",
        "step",
        "raw",
        "best_raw",
        "best_energy",
        "optimizer",
        "scheduler",
        "history",
    }
)

_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "evidence_level",
        "config",
        "environment",
        "operator",
        "operator_id",
        "initialization",
        "completed",
        "completed_steps",
        "energy",
        "energy_variance",
        "norm_error",
        "structural_antisymmetry_residual",
        "materialized_antisymmetry_residual",
        "structural_counts",
        "history",
        "refinement",
        "total_elapsed_seconds_this_call",
        "cpu_memory",
        "peak_cpu_rss_bytes",
    }
)


def validate_diagonal_path_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_operator_id: str | None = None,
    verify_operator_id: bool = False,
) -> None:
    """Validate checkpoint schema and optional resume identity constraints."""

    missing = _CHECKPOINT_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"diagonal-path checkpoint is missing fields: {sorted(missing)}")
    if payload["schema_version"] != DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported diagonal-path checkpoint schema version")
    if expected_config is not None and payload["config"] != dict(expected_config):
        raise ValueError("checkpoint configuration does not match requested run")
    if verify_operator_id and payload["operator_id"] != expected_operator_id:
        raise ValueError("checkpoint operator_id does not match requested run")
    raw = payload["raw"]
    best_raw = payload["best_raw"]
    if not isinstance(raw, torch.Tensor) or not isinstance(best_raw, torch.Tensor):
        raise ValueError("checkpoint orbitals must be tensors")
    if raw.ndim != 3 or best_raw.shape != raw.shape:
        raise ValueError("checkpoint orbitals must have matching (K,D,N) shapes")
    config = payload["config"]
    expected_shape = (
        config.get("terms"),
        config.get("basis_order"),
        config.get("particles"),
    )
    if raw.shape != expected_shape:
        raise ValueError("checkpoint orbital shape disagrees with its configuration")


def load_diagonal_path_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_config: Mapping[str, Any] | None = None,
    expected_operator_id: str | None = None,
    verify_operator_id: bool = False,
) -> dict[str, Any]:
    """Load and validate a versioned diagonal-path checkpoint."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("diagonal-path checkpoint payload must be a mapping")
    validate_diagonal_path_checkpoint(
        payload,
        expected_config=expected_config,
        expected_operator_id=expected_operator_id,
        verify_operator_id=verify_operator_id,
    )
    return dict(payload)


def validate_diagonal_path_result(
    record: Mapping[str, Any], *, require_completed: bool = False
) -> None:
    """Validate the stable scientific and reproducibility result contract."""

    missing = _RESULT_FIELDS.difference(record)
    if missing:
        raise ValueError(f"diagonal-path result is missing fields: {sorted(missing)}")
    if record["schema_version"] != DIAGONAL_PATH_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported diagonal-path result schema version")
    if record["method"] != "diagonal_path_femps":
        raise ValueError("result method is not diagonal_path_femps")
    if record["evidence_level"] != "numerical":
        raise ValueError("diagonal-path solver results must be labeled numerical")
    if require_completed and not record["completed"]:
        raise ValueError("diagonal-path result is incomplete")
    if record["structural_antisymmetry_residual"] is None:
        raise ValueError("structural antisymmetry residual must always be reported")
    counts = record["structural_counts"]
    if counts.get("enumerated_virtual_paths") != 0:
        raise ValueError("diagonal-path production may not enumerate virtual paths")
    config = record["config"]
    materialization_expected = (
        config["basis_order"] ** config["particles"]
        <= config["particle_tensor_maximum_coefficients"]
        and record["completed"]
    )
    if materialization_expected and record["materialized_antisymmetry_residual"] is None:
        raise ValueError("bounded materialization requires an antisymmetry residual")
    if record["peak_cpu_rss_bytes"] <= 0:
        raise ValueError("result must report sampled process peak RSS")
