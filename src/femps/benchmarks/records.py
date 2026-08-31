"""Stable records for controlled soft-Coulomb benchmark comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class DirectExteriorBudget:
    """Feasibility estimate for the independent dense exterior truth route."""

    particles: int
    basis_order: int
    exterior_dimension: int
    dense_complex128_bytes: int
    feasible: bool
    limit: int

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def direct_exterior_feasibility(
    particles: int,
    basis_order: int,
    *,
    maximum_dimension: int = 1200,
) -> DirectExteriorBudget:
    """Assess the committed dense-truth boundary without running an eigensolve."""

    if particles < 1 or basis_order < particles:
        raise ValueError("require 1 <= particles <= basis_order")
    if maximum_dimension < 1:
        raise ValueError("maximum_dimension must be positive")
    dimension = comb(basis_order, particles)
    return DirectExteriorBudget(
        particles=particles,
        basis_order=basis_order,
        exterior_dimension=dimension,
        dense_complex128_bytes=16 * dimension**2,
        feasible=dimension <= maximum_dimension,
        limit=maximum_dimension,
    )


@dataclass(frozen=True, slots=True)
class SoftCoulombBenchmarkPoint:
    """Common comparison fields for one optimized finite-AGP point."""

    point_id: str
    method: Literal["finite_agp"]
    evidence_level: Literal["numerical"]
    particles: int
    basis_order: int
    correlation_terms: int
    quadrature_order: int
    seed: int
    initialization: str
    energy: float
    finite_basis_reference_energy: float | None
    correlation_optimizer_error: float | None
    direct_dense_same_basis_reference_energy: float | None
    largest_basis_reference_energy: float | None
    basis_error_estimate: float | None
    total_error_estimate: float | None
    operator_error_estimate: float | None
    finite_basis_ground_fidelity: float | None
    polynomial_exterior_absolute_difference: float
    antisymmetry_residual: float | None
    retained_rank: int
    discarded_rank: int
    balanced_overlap_condition_number: float
    raw_overlap_condition_number: float | None
    pruning_or_restart_events: tuple[dict[str, Any], ...]
    elapsed_seconds: float
    peak_cuda_memory_bytes: int | None
    device: str

    def validate(self) -> None:
        if not self.point_id:
            raise ValueError("point_id must be nonempty")
        if min(
            self.particles,
            self.basis_order,
            self.correlation_terms,
            self.quadrature_order,
        ) < 1:
            raise ValueError("N, D, K, and Q must be positive")
        if self.basis_order < self.particles:
            raise ValueError("basis_order must be at least particles")
        if not 0 <= self.discarded_rank < self.correlation_terms:
            raise ValueError("discarded_rank is inconsistent with K")
        if self.retained_rank + self.discarded_rank != self.correlation_terms:
            raise ValueError("retained and discarded ranks must sum to K")
        if self.balanced_overlap_condition_number < 1:
            raise ValueError("balanced condition number must be at least one")
        if self.polynomial_exterior_absolute_difference < 0:
            raise ValueError("exterior difference must be nonnegative")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed time must be nonnegative")
        error_components = (
            self.correlation_optimizer_error,
            self.operator_error_estimate,
            self.basis_error_estimate,
            self.total_error_estimate,
        )
        if all(value is not None for value in error_components):
            correlation, operator, basis, total = error_components
            assert correlation is not None
            assert operator is not None
            assert basis is not None
            assert total is not None
            closure = abs(correlation + operator + basis - total)
            tolerance = 1e-12 + 1e-10 * abs(total)
            if closure > tolerance:
                raise ValueError("error-axis decomposition does not close")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["pruning_or_restart_events"] = list(
            self.pruning_or_restart_events
        )
        return payload


def soft_coulomb_point_from_training(
    point_id: str,
    training: dict[str, Any],
    conditioning: dict[str, Any],
    *,
    largest_basis_reference_energy: float | None = None,
    direct_dense_same_basis_reference_energy: float | None = None,
    operator_error_estimate: float | None = None,
) -> SoftCoulombBenchmarkPoint:
    """Normalize a training artifact into the Phase 12 comparison schema."""

    config = training["config"]
    if config["interaction_model"] != "soft_coulomb":
        raise ValueError("training artifact is not soft Coulomb")
    finite_reference = training.get("finite_basis_reference_energy")
    energy = float(training["final_energy"])
    correlation_error = (
        energy - float(finite_reference) if finite_reference is not None else None
    )
    direct_reference = direct_dense_same_basis_reference_energy
    basis_reference = (
        direct_reference
        if direct_reference is not None
        else (float(finite_reference) if finite_reference is not None else None)
    )
    basis_error = (
        basis_reference
        - largest_basis_reference_energy
        - (
            operator_error_estimate
            if direct_reference is None and operator_error_estimate is not None
            else 0.0
        )
        if basis_reference is not None
        and largest_basis_reference_energy is not None
        else None
    )
    operator_error = (
        float(finite_reference) - direct_reference
        if finite_reference is not None and direct_reference is not None
        else operator_error_estimate
    )
    total_error = (
        energy - largest_basis_reference_energy
        if largest_basis_reference_energy is not None
        else None
    )
    point = SoftCoulombBenchmarkPoint(
        point_id=point_id,
        method="finite_agp",
        evidence_level="numerical",
        particles=int(config["particles"]),
        basis_order=int(config["basis_order"]),
        correlation_terms=int(config["agp_terms"]),
        quadrature_order=int(config["soft_coulomb_quadrature_order"]),
        seed=int(config["seed"]),
        initialization=str(training["initialization"]),
        energy=energy,
        finite_basis_reference_energy=(
            float(finite_reference) if finite_reference is not None else None
        ),
        correlation_optimizer_error=correlation_error,
        direct_dense_same_basis_reference_energy=direct_reference,
        largest_basis_reference_energy=largest_basis_reference_energy,
        basis_error_estimate=basis_error,
        total_error_estimate=total_error,
        operator_error_estimate=operator_error,
        finite_basis_ground_fidelity=training.get("finite_basis_ground_fidelity"),
        polynomial_exterior_absolute_difference=float(
            training["polynomial_explicit_absolute_difference"]
        ),
        antisymmetry_residual=training.get("antisymmetry_residual"),
        retained_rank=int(conditioning["retained_rank"]),
        discarded_rank=int(conditioning["discarded_rank"]),
        balanced_overlap_condition_number=float(
            conditioning["balanced_overlap_condition_number"]
        ),
        raw_overlap_condition_number=conditioning.get(
            "raw_overlap_condition_number"
        ),
        pruning_or_restart_events=tuple(
            conditioning.get("pruning_or_restart_events", [])
        ),
        elapsed_seconds=float(training["elapsed_seconds_this_call"]),
        peak_cuda_memory_bytes=training.get("peak_cuda_memory_bytes"),
        device=str(training["environment"]["device"]),
    )
    point.validate()
    return point
