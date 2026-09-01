"""Truth-free adaptive determinant growth for diagonal-path FEMPS.

Candidate terms are drawn from a deterministic seeded pool and ranked only by
the polynomial determinant-transition generalized eigenproblem.  Dense CI
vectors, CI energies, and materialized particle tensors are not inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from femps.exterior import diagonal_path_hamiltonian_matrices
from femps.hamiltonians import FactorizedTwoBodyOperator

from .agp_subspace import solve_generalized_hermitian
from .diagonal_path_training import canonical_slater_orbitals


@dataclass(frozen=True, slots=True)
class AdaptiveTermCandidate:
    """Fixed-span diagnostic for one seeded Slater candidate."""

    candidate_index: int
    predicted_energy: float
    predicted_improvement: float
    retained_rank: int
    discarded_rank: int
    retained_condition_number: float
    admitted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class AdaptiveTermGrowth:
    """Selected one-term extension and its auditable candidate diagnostics."""

    orbitals: torch.Tensor
    seed: int
    pool_size: int
    source_terms: int
    selected_candidate: int
    source_energy: float
    predicted_energy: float
    predicted_improvement: float
    candidates: tuple[AdaptiveTermCandidate, ...]


def _seeded_candidate_pool(
    source: torch.Tensor, *, pool_size: int, seed: int
) -> torch.Tensor:
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    shape = (pool_size, source.shape[1], source.shape[2])
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    if source.is_complex():
        imaginary = torch.randn(shape, generator=generator, dtype=torch.float64)
        raw = torch.complex(real, imaginary)
    else:
        raw = real
    return canonical_slater_orbitals(
        raw.to(dtype=source.dtype, device=source.device)
    )


@torch.no_grad()
def select_adaptive_diagonal_path_term(
    orbitals: torch.Tensor,
    one_body: torch.Tensor,
    interaction: FactorizedTwoBodyOperator | None = None,
    *,
    pool_size: int,
    seed: int,
    overlap_relative_threshold: float = 1e-10,
    condition_threshold: float = 1e8,
    energy_nesting_tolerance: float = 1e-10,
) -> AdaptiveTermGrowth:
    """Select one blind Slater by its fixed-span variational improvement.

    The source state is first QR gauged.  Every candidate augments that exact
    nonlinear span, and its linear amplitudes are eliminated with the same
    conditioned generalized eigenproblem as production training.  A candidate
    is admitted only when all ``K+1`` directions survive and the balanced
    overlap condition is below ``condition_threshold``.  Ties are resolved by
    the seeded pool order.
    """

    if orbitals.ndim != 3 or orbitals.shape[1] < orbitals.shape[2]:
        raise ValueError("orbitals must have shape (K,D,N) with D >= N")
    if one_body.shape != (orbitals.shape[1], orbitals.shape[1]):
        raise ValueError("one_body has the wrong shape")
    if interaction is not None and interaction.dimension != orbitals.shape[1]:
        raise ValueError("interaction has the wrong dimension")
    if overlap_relative_threshold < 0:
        raise ValueError("overlap_relative_threshold must be nonnegative")
    if condition_threshold < 1:
        raise ValueError("condition_threshold must be at least one")
    if energy_nesting_tolerance < 0:
        raise ValueError("energy_nesting_tolerance must be nonnegative")

    source = canonical_slater_orbitals(orbitals)
    factor_arguments = {
        "two_body_left": interaction.left if interaction is not None else None,
        "two_body_right": interaction.right if interaction is not None else None,
        "two_body_weights": interaction.weights if interaction is not None else None,
    }
    source_overlap, source_hamiltonian = diagonal_path_hamiltonian_matrices(
        source, one_body, **factor_arguments
    )
    source_solved = solve_generalized_hermitian(
        source_hamiltonian,
        source_overlap,
        relative_threshold=overlap_relative_threshold,
    )
    if source_solved.retained_rank != source.shape[0]:
        raise ValueError("source determinant span is rank deficient")
    if source_solved.retained_condition_number > condition_threshold:
        raise ValueError("source determinant span exceeds the condition threshold")
    source_energy = float(source_solved.energy.detach().cpu())

    pool = _seeded_candidate_pool(source, pool_size=pool_size, seed=seed)
    diagnostics: list[AdaptiveTermCandidate] = []
    admitted: list[tuple[float, int, torch.Tensor]] = []
    target_rank = source.shape[0] + 1
    for index in range(pool_size):
        extended = torch.cat((source, pool[index : index + 1]), dim=0)
        overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
            extended, one_body, **factor_arguments
        )
        solved = solve_generalized_hermitian(
            hamiltonian,
            overlap,
            relative_threshold=overlap_relative_threshold,
        )
        energy = float(solved.energy.detach().cpu())
        rank_pass = solved.retained_rank == target_rank
        condition_pass = solved.retained_condition_number <= condition_threshold
        nesting_pass = energy <= source_energy + energy_nesting_tolerance
        is_admitted = rank_pass and condition_pass and nesting_pass
        if not rank_pass:
            reason = "augmented span loses a balanced-overlap direction"
        elif not condition_pass:
            reason = "augmented span exceeds the condition threshold"
        elif not nesting_pass:
            reason = "augmented energy violates the nesting tolerance"
        else:
            reason = "admitted"
            admitted.append((energy, index, extended))
        diagnostics.append(
            AdaptiveTermCandidate(
                candidate_index=index,
                predicted_energy=energy,
                predicted_improvement=source_energy - energy,
                retained_rank=solved.retained_rank,
                discarded_rank=solved.discarded_rank,
                retained_condition_number=solved.retained_condition_number,
                admitted=is_admitted,
                reason=reason,
            )
        )

    if not admitted:
        raise RuntimeError("no adaptive growth candidate passed the registered gates")
    predicted_energy, selected_index, selected_orbitals = min(
        admitted, key=lambda item: (item[0], item[1])
    )
    return AdaptiveTermGrowth(
        orbitals=selected_orbitals.clone(),
        seed=seed,
        pool_size=pool_size,
        source_terms=source.shape[0],
        selected_candidate=selected_index,
        source_energy=source_energy,
        predicted_energy=predicted_energy,
        predicted_improvement=source_energy - predicted_energy,
        candidates=tuple(diagnostics),
    )


__all__ = [
    "AdaptiveTermCandidate",
    "AdaptiveTermGrowth",
    "select_adaptive_diagonal_path_term",
]
