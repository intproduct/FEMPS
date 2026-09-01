"""End-to-end clean Slater-source orchestration for restricted FEMPS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from femps.benchmarks import ProcessRSSMonitor
from femps.hamiltonians import harmonic_pair_hamiltonian, soft_coulomb_operator

from .adaptive_diagonal_path_training import (
    AdaptiveDiagonalPathConfig,
    AdaptiveDiagonalPathStageConfig,
    run_bounded_adaptive_diagonal_path,
)
from .diagonal_path_contract import load_diagonal_path_checkpoint
from .diagonal_path_training import (
    DiagonalPathConfig,
    canonical_slater_orbitals,
    run_diagonal_path_variable_projection,
)
from .slater_source_contract import (
    SLATER_SOURCE_CHECKPOINT_SCHEMA_VERSION,
    SLATER_SOURCE_METHOD,
    SLATER_SOURCE_RESULT_SCHEMA_VERSION,
    load_slater_source_checkpoint,
    validate_slater_source_result,
)


SCIENTIFIC_BOUNDARY = (
    "clean single-Slater-source command for restricted nonbranching first-quantized "
    "continuous functional-basis FEMPS; bounded numerical evidence; no automatic stopping"
)


@dataclass(frozen=True, slots=True)
class SlaterSourceOptimizerConfig:
    seed: int
    steps: int
    learning_rate: float
    final_learning_rate: float
    record_points: int
    checkpoint_every: int
    lbfgs_refinement_steps: int
    lbfgs_learning_rate: float

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("optimizer seed must be nonnegative")
        if min(self.steps, self.record_points, self.checkpoint_every) < 1:
            raise ValueError("optimizer step and record counts must be positive")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError("invalid optimizer learning-rate schedule")
        if self.lbfgs_refinement_steps < 0 or self.lbfgs_learning_rate <= 0:
            raise ValueError("invalid L-BFGS configuration")


@dataclass(frozen=True, slots=True)
class SlaterSourceSolverConfig:
    particles: int
    basis_order: int
    device: str
    omega: float
    coupling: float
    softening: float
    quadrature_order: int
    relative_factor_threshold: float
    factorization_backend: str
    source_optimizer: SlaterSourceOptimizerConfig
    stage_optimizer: SlaterSourceOptimizerConfig
    max_terms: int
    pool_size: int
    stages: tuple[AdaptiveDiagonalPathStageConfig, ...]
    overlap_relative_threshold: float
    condition_threshold: float
    energy_nesting_tolerance: float
    truth_maximum_dimension: int
    particle_tensor_maximum_coefficients: int

    def validate(self) -> None:
        if self.particles < 1 or self.basis_order < self.particles:
            raise ValueError("require D >= N >= 1")
        if self.device != "cpu":
            raise ValueError("clean Slater-source orchestration currently requires CPU")
        if self.omega <= 0 or self.coupling < 0 or self.softening <= 0:
            raise ValueError("invalid physical model parameters")
        if self.quadrature_order < 1 or self.relative_factor_threshold < 0:
            raise ValueError("invalid soft-Coulomb factorization parameters")
        if self.factorization_backend != "physical":
            raise ValueError("Phase 37 admits only the physical-SVD backend")
        if min(
            self.truth_maximum_dimension,
            self.particle_tensor_maximum_coefficients,
        ) < 1:
            raise ValueError("truth/materialization caps must be positive")
        self.source_optimizer.validate()
        self.stage_optimizer.validate()
        self.adaptive_config().validate(1)

    def adaptive_config(self) -> AdaptiveDiagonalPathConfig:
        return AdaptiveDiagonalPathConfig(
            max_terms=self.max_terms,
            pool_size=self.pool_size,
            stages=self.stages,
            overlap_relative_threshold=self.overlap_relative_threshold,
            condition_threshold=self.condition_threshold,
            energy_nesting_tolerance=self.energy_nesting_tolerance,
        )


def _optimizer_record(
    record: dict[str, Any], label: str, *, default_seed: int | None = None
) -> SlaterSourceOptimizerConfig:
    try:
        values = dict(record)
        if default_seed is not None:
            values.setdefault("seed", default_seed)
        return SlaterSourceOptimizerConfig(**values)
    except TypeError as error:
        raise ValueError(f"invalid {label} optimizer record") from error


def slater_source_config_from_record(record: dict[str, Any]) -> SlaterSourceSolverConfig:
    """Parse and validate the versioned JSON command configuration."""

    if record.get("schema_version") != 1:
        raise ValueError("unsupported Slater-source command config schema")
    if record.get("method") != SLATER_SOURCE_METHOD:
        raise ValueError("unexpected Slater-source command config method")
    if record.get("evidence_level") != "numerical":
        raise ValueError("command config must be labeled numerical evidence")
    if record.get("model", {}).get("interaction") != "soft_coulomb":
        raise ValueError("clean command currently supports soft_coulomb only")
    model = record["model"]
    adaptive = record["adaptive"]
    validation = record["validation"]
    stages = tuple(
        AdaptiveDiagonalPathStageConfig(**stage) for stage in adaptive["stages"]
    )
    config = SlaterSourceSolverConfig(
        particles=int(record["particles"]),
        basis_order=int(record["basis_order"]),
        device=str(record["device"]),
        omega=float(model["omega"]),
        coupling=float(model["coupling"]),
        softening=float(model["softening"]),
        quadrature_order=int(model["quadrature_order"]),
        relative_factor_threshold=float(model["relative_factor_threshold"]),
        factorization_backend=str(model["factorization_backend"]),
        source_optimizer=_optimizer_record(
            record["source_optimizer"], "source"
        ),
        stage_optimizer=_optimizer_record(
            record["stage_optimizer"], "stage", default_seed=0
        ),
        max_terms=int(adaptive["max_terms"]),
        pool_size=int(adaptive["pool_size"]),
        stages=stages,
        overlap_relative_threshold=float(adaptive["overlap_relative_threshold"]),
        condition_threshold=float(adaptive["condition_threshold"]),
        energy_nesting_tolerance=float(adaptive["energy_nesting_tolerance"]),
        truth_maximum_dimension=int(validation["truth_maximum_dimension"]),
        particle_tensor_maximum_coefficients=int(
            validation["particle_tensor_maximum_coefficients"]
        ),
    )
    config.validate()
    return config


def load_slater_source_command_config(
    path: Path,
) -> tuple[SlaterSourceSolverConfig, dict[str, Any]]:
    """Load a solver config and retain its explicit paths/acceptance record."""

    record = json.loads(path.read_text(encoding="utf-8"))
    config = slater_source_config_from_record(record)
    for field in ("checkpoint_path", "output_path", "acceptance"):
        if field not in record:
            raise ValueError(f"command config is missing {field}")
    return config, record


def canonical_lowest_slater(config: SlaterSourceSolverConfig) -> torch.Tensor:
    """Construct the canonical determinant occupying the lowest N basis states."""

    source = torch.zeros(
        (1, config.basis_order, config.particles), dtype=torch.complex128
    )
    source[0, : config.particles, :] = torch.eye(
        config.particles, dtype=torch.complex128
    )
    return canonical_slater_orbitals(source)


def _hash_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _operator_hash(one_body: torch.Tensor, interaction: Any) -> str:
    digest = hashlib.sha256()
    for tensor in (one_body, interaction.left, interaction.right, interaction.weights):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _config_record(config: SlaterSourceSolverConfig) -> dict[str, Any]:
    return asdict(config)


def _config_hash(record: dict[str, Any]) -> str:
    serialized = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _child_path(checkpoint_path: Path, label: str) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.{label}.pt")


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _optimizer_config(
    config: SlaterSourceSolverConfig,
    optimizer: SlaterSourceOptimizerConfig,
    *,
    terms: int,
) -> DiagonalPathConfig:
    return DiagonalPathConfig(
        basis_order=config.basis_order,
        particles=config.particles,
        terms=terms,
        omega=config.omega,
        interaction_model="soft_coulomb",
        soft_coulomb_coupling=config.coupling,
        soft_coulomb_softening=config.softening,
        soft_coulomb_quadrature_order=config.quadrature_order,
        soft_coulomb_relative_threshold=config.relative_factor_threshold,
        steps=optimizer.steps,
        learning_rate=optimizer.learning_rate,
        final_learning_rate=optimizer.final_learning_rate,
        seed=optimizer.seed,
        device=config.device,
        record_points=optimizer.record_points,
        checkpoint_every=optimizer.checkpoint_every,
        overlap_relative_threshold=config.overlap_relative_threshold,
        truth_maximum_dimension=config.truth_maximum_dimension,
        particle_tensor_maximum_coefficients=config.particle_tensor_maximum_coefficients,
        lbfgs_refinement_steps=optimizer.lbfgs_refinement_steps,
        lbfgs_learning_rate=optimizer.lbfgs_learning_rate,
    )


def _stage_records(source_result: dict[str, Any], adaptive_result: dict | None) -> list[dict]:
    stages = [
        {
            "terms": 1,
            "role": "optimized_canonical_slater_source",
            "optimizer_result": source_result,
        }
    ]
    if adaptive_result is not None:
        stages.extend(
            {
                "terms": stage["target_terms"],
                "role": "bounded_adaptive_growth",
                "selected_candidate": stage["selected_candidate"],
                "candidate_seed": stage["candidate_seed"],
                "optimizer_seed": stage["optimizer_seed"],
                "predicted_improvement": stage["predicted_improvement"],
                "optimizer_result": stage["optimizer_result"],
            }
            for stage in adaptive_result["stages"]
        )
    return stages


def run_slater_source_adaptive_solver(
    config: SlaterSourceSolverConfig,
    *,
    checkpoint_path: Path,
    resume: bool = False,
    max_source_steps_this_call: int | None = None,
    max_adaptive_stages_this_call: int | None = None,
) -> dict[str, Any]:
    """Run K1 optimization and bounded adaptive growth from no prior FEMPS state."""

    config.validate()
    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path")
    initial_source = canonical_lowest_slater(config)
    one_body = harmonic_pair_hamiltonian(
        config.basis_order,
        kappa=0.0,
        omega=config.omega,
        dtype=torch.complex128,
        device="cpu",
    )[0]
    interaction, diagnostics = soft_coulomb_operator(
        config.basis_order,
        quadrature_order=config.quadrature_order,
        coupling=config.coupling,
        softening=config.softening,
        relative_threshold=config.relative_factor_threshold,
        factorization_backend=config.factorization_backend,
        dtype=torch.complex128,
        device="cpu",
    )
    config_record = _config_record(config)
    config_sha256 = _config_hash(config_record)
    initial_source_identity = {
        "source_id": "canonical_lowest_functional_basis_slater",
        "orbitals_sha256": _hash_tensor(initial_source),
        "shape": list(initial_source.shape),
    }
    operator_identity = {
        "operator_id": (
            f"soft_coulomb_N{config.particles}_D{config.basis_order}_"
            f"Q{config.quadrature_order}_{config.factorization_backend}_svd"
        ),
        "operator_sha256": _operator_hash(one_body, interaction),
        "dimension": config.basis_order,
        "factor_rank": interaction.rank,
    }
    operator_metadata = {
        "model": "soft_coulomb",
        "omega": config.omega,
        "coupling": config.coupling,
        "softening": config.softening,
        "quadrature_order": config.quadrature_order,
        "factorization_backend": diagnostics.factorization_backend,
        "factor_rank": diagnostics.retained_rank,
        "relative_factor_threshold": diagnostics.relative_threshold,
        "dense_relative_factorization_error": (
            diagnostics.dense_relative_factorization_error
        ),
    }
    source_checkpoint = _child_path(checkpoint_path, "source.optimizer")
    adaptive_checkpoint = _child_path(checkpoint_path, "adaptive")
    adaptive_optimizer_paths = [
        adaptive_checkpoint.with_name(
            f"{adaptive_checkpoint.stem}.K{stage.target_terms}.optimizer.pt"
        )
        for stage in config.stages
    ]
    source_config = _optimizer_config(config, config.source_optimizer, terms=1)
    stage_template = _optimizer_config(config, config.stage_optimizer, terms=1)

    if resume:
        if not checkpoint_path.is_file():
            raise ValueError("resume requires an existing command checkpoint")
        checkpoint = load_slater_source_checkpoint(
            checkpoint_path,
            expected_config=config_record,
            expected_initial_source_identity=initial_source_identity,
            expected_operator_identity=operator_identity,
        )
        source_result = checkpoint["source_result"]
        accepted_source_raw = checkpoint["accepted_source_raw"]
        adaptive_result = checkpoint["adaptive_result"]
    else:
        stale = [
            path
            for path in (
                checkpoint_path,
                source_checkpoint,
                adaptive_checkpoint,
                *adaptive_optimizer_paths,
            )
            if path.exists()
        ]
        if stale:
            raise ValueError(f"command checkpoint lineage already exists: {stale[0]}")
        source_result = None
        accepted_source_raw = None
        adaptive_result = None

    total_started = time.perf_counter()
    with ProcessRSSMonitor() as memory_monitor:
        if source_result is None or not source_result["completed"]:
            source_result = run_diagonal_path_variable_projection(
                source_config,
                checkpoint_path=source_checkpoint,
                resume=resume and source_checkpoint.is_file(),
                max_steps_this_call=max_source_steps_this_call,
                initial_orbitals=(
                    None if resume and source_checkpoint.is_file() else initial_source
                ),
                operators=(one_body, interaction),
                operator_id=operator_identity["operator_id"],
            )
            if source_result["completed"]:
                accepted_source_raw = load_diagonal_path_checkpoint(source_checkpoint)[
                    "best_raw"
                ]

        if source_result["completed"]:
            adaptive_result = run_bounded_adaptive_diagonal_path(
                accepted_source_raw,
                one_body,
                interaction,
                stage_template,
                config.adaptive_config(),
                source_id="optimized_canonical_lowest_functional_basis_slater",
                operator_id=operator_identity["operator_id"],
                checkpoint_path=adaptive_checkpoint,
                resume=adaptive_checkpoint.is_file(),
                max_stages_this_call=max_adaptive_stages_this_call,
            )
    memory_record = memory_monitor.record()
    source_completed = bool(source_result["completed"])
    completed = bool(source_completed and adaptive_result and adaptive_result["completed"])
    current_terms = int(adaptive_result["current_terms"] if adaptive_result else 1)
    accepted_source_identity = (
        {
            "source_id": "optimized_canonical_lowest_functional_basis_slater",
            "orbitals_sha256": _hash_tensor(
                canonical_slater_orbitals(accepted_source_raw)
            ),
            "shape": list(accepted_source_raw.shape),
        }
        if source_completed
        else None
    )
    checkpoint_record = {
        "schema_version": SLATER_SOURCE_CHECKPOINT_SCHEMA_VERSION,
        "method": SLATER_SOURCE_METHOD,
        "evidence_level": "numerical",
        "scientific_boundary": SCIENTIFIC_BOUNDARY,
        "config": config_record,
        "config_sha256": config_sha256,
        "initial_source_identity": initial_source_identity,
        "operator_identity": operator_identity,
        "source_completed": source_completed,
        "source_result": source_result,
        "accepted_source_raw": (
            accepted_source_raw.detach().cpu() if source_completed else None
        ),
        "adaptive_result": adaptive_result,
        "current_terms": current_terms,
        "completed": completed,
    }
    _save_checkpoint(checkpoint_path, checkpoint_record)
    result = {
        "schema_version": SLATER_SOURCE_RESULT_SCHEMA_VERSION,
        "method": SLATER_SOURCE_METHOD,
        "evidence_level": "numerical",
        "scientific_boundary": SCIENTIFIC_BOUNDARY,
        "config": config_record,
        "config_sha256": config_sha256,
        "source_construction": {
            "kind": "canonical_lowest_functional_basis_slater",
            "historical_checkpoint_used": False,
            "ci_initializer_used": False,
            "source_seed": config.source_optimizer.seed,
        },
        "initial_source_identity": initial_source_identity,
        "accepted_source_identity": accepted_source_identity,
        "operator_identity": operator_identity,
        "operator_metadata": operator_metadata,
        "source_result": source_result,
        "adaptive_result": adaptive_result,
        "stages": _stage_records(source_result, adaptive_result),
        "current_terms": current_terms,
        "completed": completed,
        "resumed": resume,
        "total_elapsed_seconds_this_call": time.perf_counter() - total_started,
        "cpu_memory": memory_record.as_dict(),
        "peak_cpu_rss_bytes": memory_record.peak_rss_bytes,
        "peak_cpu_rss_delta_bytes": memory_record.peak_delta_rss_bytes,
        "automatic_stopping_rule": "not_admitted",
        "external_max_terms_required": True,
    }
    validate_slater_source_result(result)
    return result


__all__ = [
    "SCIENTIFIC_BOUNDARY",
    "SlaterSourceOptimizerConfig",
    "SlaterSourceSolverConfig",
    "canonical_lowest_slater",
    "load_slater_source_command_config",
    "run_slater_source_adaptive_solver",
    "slater_source_config_from_record",
]
