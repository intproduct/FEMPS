"""Bounded, checkpointed adaptive correlation growth for diagonal-path FEMPS."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

import torch

from femps.hamiltonians import FactorizedTwoBodyOperator

from .adaptive_diagonal_path_contract import (
    ADAPTIVE_DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION,
    ADAPTIVE_DIAGONAL_PATH_RESULT_SCHEMA_VERSION,
    load_adaptive_diagonal_path_checkpoint,
    validate_adaptive_diagonal_path_result,
)
from .diagonal_path_contract import load_diagonal_path_checkpoint
from .diagonal_path_growth import AdaptiveTermGrowth, select_adaptive_diagonal_path_term
from .diagonal_path_training import (
    DiagonalPathConfig,
    canonical_slater_orbitals,
    run_diagonal_path_variable_projection,
)


SCIENTIFIC_BOUNDARY = (
    "restricted nonbranching first-quantized continuous functional-basis FEMPS; "
    "bounded numerical evidence; automatic stopping is not admitted"
)


@dataclass(frozen=True, slots=True)
class AdaptiveDiagonalPathStageConfig:
    """Caller-supplied deterministic seeds for one target correlation count."""

    target_terms: int
    candidate_seed: int
    optimizer_seed: int


@dataclass(frozen=True, slots=True)
class AdaptiveDiagonalPathConfig:
    """Explicit finite adaptive schedule with a mandatory external maximum K."""

    max_terms: int
    pool_size: int
    stages: tuple[AdaptiveDiagonalPathStageConfig, ...]
    overlap_relative_threshold: float = 1e-10
    condition_threshold: float = 1e8
    energy_nesting_tolerance: float = 1e-10

    def validate(self, start_terms: int) -> None:
        if start_terms < 1:
            raise ValueError("source K must be positive")
        if self.max_terms <= start_terms:
            raise ValueError("external max_terms must be greater than source K")
        if self.pool_size < 1:
            raise ValueError("candidate pool_size must be positive")
        if self.overlap_relative_threshold < 0:
            raise ValueError("overlap threshold must be nonnegative")
        if self.condition_threshold < 1:
            raise ValueError("condition threshold must be at least one")
        if self.energy_nesting_tolerance < 0:
            raise ValueError("nesting tolerance must be nonnegative")
        targets = [stage.target_terms for stage in self.stages]
        expected = list(range(start_terms + 1, self.max_terms + 1))
        if targets != expected:
            raise ValueError(
                "stage schedule must explicitly cover every K through max_terms"
            )
        if any(stage.candidate_seed < 0 or stage.optimizer_seed < 0 for stage in self.stages):
            raise ValueError("adaptive seeds must be nonnegative")


def _hash_tensor(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    _hash_tensor(digest, tensor)
    return digest.hexdigest()


def _operator_sha256(
    one_body: torch.Tensor, interaction: FactorizedTwoBodyOperator | None
) -> str:
    digest = hashlib.sha256()
    _hash_tensor(digest, one_body)
    if interaction is None:
        digest.update(b"no_two_body_operator")
    else:
        _hash_tensor(digest, interaction.left)
        _hash_tensor(digest, interaction.right)
        _hash_tensor(digest, interaction.weights)
    return digest.hexdigest()


def _growth_record(growth: AdaptiveTermGrowth) -> dict[str, Any]:
    return {
        "candidate_seed": growth.seed,
        "pool_size": growth.pool_size,
        "source_terms": growth.source_terms,
        "selected_candidate": growth.selected_candidate,
        "source_energy": growth.source_energy,
        "predicted_energy": growth.predicted_energy,
        "predicted_improvement": growth.predicted_improvement,
        "truth_state_used": False,
        "candidates": [asdict(candidate) for candidate in growth.candidates],
    }


def _stage_optimizer_checkpoint(checkpoint_path: Path, target_terms: int) -> Path:
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}.K{target_terms}.optimizer.pt"
    )


def _save_outer_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _identity_records(
    source: torch.Tensor,
    one_body: torch.Tensor,
    interaction: FactorizedTwoBodyOperator | None,
    *,
    source_id: str,
    operator_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not source_id or not operator_id:
        raise ValueError("source_id and operator_id must be nonempty")
    return (
        {
            "source_id": source_id,
            "orbitals_sha256": _tensor_sha256(source),
            "shape": list(source.shape),
        },
        {
            "operator_id": operator_id,
            "operator_sha256": _operator_sha256(one_body, interaction),
            "dimension": int(one_body.shape[0]),
            "factor_rank": interaction.rank if interaction is not None else 0,
        },
    )


def run_bounded_adaptive_diagonal_path(
    source_orbitals: torch.Tensor,
    one_body: torch.Tensor,
    interaction: FactorizedTwoBodyOperator | None,
    optimizer_template: DiagonalPathConfig,
    adaptive_config: AdaptiveDiagonalPathConfig,
    *,
    source_id: str,
    operator_id: str,
    checkpoint_path: Path,
    resume: bool = False,
    max_stages_this_call: int | None = None,
) -> dict[str, Any]:
    """Run an explicit finite K-growth schedule and checkpoint every stage.

    This function never decides its own final K. The caller must provide every
    candidate and optimizer seed through ``adaptive_config.stages`` and must
    set a finite ``max_terms`` greater than the source term count.
    """

    if source_orbitals.ndim != 3 or source_orbitals.shape[1] < source_orbitals.shape[2]:
        raise ValueError("source_orbitals must have shape (K,D,N) with D >= N")
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path")
    if max_stages_this_call is not None and max_stages_this_call < 1:
        raise ValueError("max_stages_this_call must be positive")
    source = canonical_slater_orbitals(
        source_orbitals.to(dtype=torch.complex128, device="cpu")
    )
    start_terms, dimension, particles = source.shape
    optimizer_template.validate()
    if (
        optimizer_template.terms != start_terms
        or optimizer_template.basis_order != dimension
        or optimizer_template.particles != particles
    ):
        raise ValueError("optimizer template must describe the source K,D,N")
    if optimizer_template.device != "cpu":
        raise ValueError("bounded adaptive orchestration currently requires CPU")
    if one_body.shape != (dimension, dimension):
        raise ValueError("one_body has the wrong shape")
    if interaction is not None and interaction.dimension != dimension:
        raise ValueError("interaction has the wrong dimension")
    adaptive_config.validate(start_terms)

    source_identity, operator_identity = _identity_records(
        source,
        one_body,
        interaction,
        source_id=source_id,
        operator_id=operator_id,
    )
    adaptive_record = asdict(adaptive_config)
    optimizer_record = asdict(optimizer_template)

    if resume:
        if not checkpoint_path.is_file():
            raise ValueError("resume requires an existing adaptive checkpoint")
        checkpoint = load_adaptive_diagonal_path_checkpoint(
            checkpoint_path,
            expected_source_identity=source_identity,
            expected_operator_identity=operator_identity,
            expected_adaptive_config=adaptive_record,
            expected_optimizer_template=optimizer_record,
        )
        current = checkpoint["current_orbitals"].to(dtype=torch.complex128)
        stages = list(checkpoint["stages"])
        current_terms = int(checkpoint["current_terms"])
    else:
        if checkpoint_path.exists():
            raise ValueError("adaptive checkpoint already exists; pass resume=True")
        current = source
        stages = []
        current_terms = start_terms

    remaining = [
        stage for stage in adaptive_config.stages if stage.target_terms > current_terms
    ]
    if max_stages_this_call is not None:
        remaining = remaining[:max_stages_this_call]

    stages_this_call = 0
    for stage_config in remaining:
        growth = select_adaptive_diagonal_path_term(
            current,
            one_body,
            interaction,
            pool_size=adaptive_config.pool_size,
            seed=stage_config.candidate_seed,
            overlap_relative_threshold=adaptive_config.overlap_relative_threshold,
            condition_threshold=adaptive_config.condition_threshold,
            energy_nesting_tolerance=adaptive_config.energy_nesting_tolerance,
        )
        nesting_error = float(
            torch.max(torch.abs(growth.orbitals[:current_terms] - current)).cpu()
        )
        stage_optimizer_config = replace(
            optimizer_template,
            terms=stage_config.target_terms,
            seed=stage_config.optimizer_seed,
        )
        optimizer_checkpoint = _stage_optimizer_checkpoint(
            checkpoint_path, stage_config.target_terms
        )
        optimizer_result = run_diagonal_path_variable_projection(
            stage_optimizer_config,
            checkpoint_path=optimizer_checkpoint,
            initial_orbitals=growth.orbitals,
            operators=(one_body, interaction),
            operator_id=operator_id,
        )
        optimized = canonical_slater_orbitals(
            load_diagonal_path_checkpoint(optimizer_checkpoint)["best_raw"]
        )
        growth_record = _growth_record(growth)
        stages.append(
            {
                "target_terms": stage_config.target_terms,
                "candidate_seed": stage_config.candidate_seed,
                "optimizer_seed": stage_config.optimizer_seed,
                "selected_candidate": growth.selected_candidate,
                "predicted_improvement": growth.predicted_improvement,
                "source_nesting_max_abs_error": nesting_error,
                "growth": growth_record,
                "optimizer_result": optimizer_result,
                "optimized_orbitals_sha256": _tensor_sha256(optimized),
            }
        )
        current = optimized
        current_terms = stage_config.target_terms
        stages_this_call += 1
        checkpoint_record = {
            "schema_version": ADAPTIVE_DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION,
            "method": "bounded_adaptive_diagonal_path_femps",
            "evidence_level": "numerical",
            "scientific_boundary": SCIENTIFIC_BOUNDARY,
            "source_identity": source_identity,
            "operator_identity": operator_identity,
            "adaptive_config": adaptive_record,
            "optimizer_template": optimizer_record,
            "start_terms": start_terms,
            "current_terms": current_terms,
            "current_orbitals": current.detach().cpu(),
            "stages": stages,
            "completed": current_terms == adaptive_config.max_terms,
        }
        _save_outer_checkpoint(checkpoint_path, checkpoint_record)

    completed = current_terms == adaptive_config.max_terms
    if not stages:
        raise ValueError("adaptive run contains no completed growth stage")
    final_optimizer_result = stages[-1]["optimizer_result"]
    result = {
        "schema_version": ADAPTIVE_DIAGONAL_PATH_RESULT_SCHEMA_VERSION,
        "method": "bounded_adaptive_diagonal_path_femps",
        "evidence_level": "numerical",
        "scientific_boundary": SCIENTIFIC_BOUNDARY,
        "source_identity": source_identity,
        "operator_identity": operator_identity,
        "adaptive_config": adaptive_record,
        "optimizer_template": optimizer_record,
        "start_terms": start_terms,
        "current_terms": current_terms,
        "completed": completed,
        "resumed": resume,
        "stages_completed_this_call": stages_this_call,
        "stages": stages,
        "final_energy": final_optimizer_result["energy"],
        "structural_antisymmetry_residual": final_optimizer_result[
            "structural_antisymmetry_residual"
        ],
        "enumerated_virtual_paths": final_optimizer_result["structural_counts"][
            "enumerated_virtual_paths"
        ],
        "materialized_particle_coefficients": final_optimizer_result[
            "structural_counts"
        ]["materialized_particle_coefficients"],
        "automatic_stopping_rule": "not_admitted",
        "external_max_terms_required": True,
    }
    validate_adaptive_diagonal_path_result(result)
    return result


__all__ = [
    "AdaptiveDiagonalPathConfig",
    "AdaptiveDiagonalPathStageConfig",
    "SCIENTIFIC_BOUNDARY",
    "run_bounded_adaptive_diagonal_path",
]
