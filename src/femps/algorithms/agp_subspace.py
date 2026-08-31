"""Conditioned linear-amplitude solve for a fixed finite-AGP basis."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GeneralizedEigenResult:
    """Lowest generalized Hermitian eigenpair and overlap diagnostics."""

    energy: torch.Tensor
    amplitudes: torch.Tensor
    # Eigenvalues of the unit-diagonal, gauge-balanced overlap matrix.
    overlap_eigenvalues: torch.Tensor
    raw_overlap_eigenvalues: torch.Tensor
    retained_rank: int
    discarded_rank: int
    retained_condition_number: float
    raw_overlap_condition_number: float | None
    residual_norm: torch.Tensor


@dataclass(frozen=True, slots=True)
class OverlapWhitening:
    """Deterministic isometry from an orthonormal retained span to AGP terms."""

    transformation: torch.Tensor
    # Eigenvalues after balancing every nonzero term to unit norm.
    eigenvalues: torch.Tensor
    raw_eigenvalues: torch.Tensor
    term_norms: torch.Tensor
    retained_rank: int
    discarded_rank: int


@dataclass(frozen=True, slots=True)
class TermPruningAssessment:
    """Auditable decision about removing one nonlinear AGP term."""

    should_prune: bool
    candidate: int | None
    energy_penalty: float | None
    balanced_condition_number: float
    discarded_rank: int
    reason: str


def overlap_whitening(
    overlap: torch.Tensor,
    *,
    relative_threshold: float = 1e-10,
    absolute_threshold: float = 0.0,
) -> OverlapWhitening:
    """Return ``W`` with ``W^dag S W=I`` after gauge-balanced compression.

    The diagonal balancing is essential: independent rescaling of nonlinear AGP
    terms is a coordinate gauge and must not decide whether a physical span
    direction is discarded.  Thresholds therefore act on the unit-diagonal
    correlation matrix rather than on the raw overlap matrix.
    """

    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("overlap must be square")
    if relative_threshold < 0 or absolute_threshold < 0:
        raise ValueError("overlap thresholds must be nonnegative")
    hermitian = 0.5 * (overlap + overlap.mH)
    raw_eigenvalues = torch.linalg.eigvalsh(hermitian)
    diagonal = torch.diagonal(hermitian).real
    if torch.any(diagonal <= torch.finfo(diagonal.dtype).tiny):
        raise ValueError("overlap contains a zero-norm term")
    term_norms = torch.sqrt(diagonal)
    balanced = hermitian / (term_norms[:, None] * term_norms[None, :])
    balanced = 0.5 * (balanced + balanced.mH)
    eigenvalues, eigenvectors = torch.linalg.eigh(balanced)
    largest = eigenvalues[-1]
    threshold = torch.maximum(
        torch.as_tensor(
            absolute_threshold, dtype=eigenvalues.dtype, device=overlap.device
        ),
        relative_threshold * largest,
    )
    if largest <= 0:
        raise ValueError("overlap matrix has no positive direction")
    if eigenvalues[0] < -threshold:
        raise ValueError("overlap matrix has a material negative eigenvalue")
    retained = eigenvalues > threshold
    rank = int(torch.count_nonzero(retained))
    if rank == 0:
        raise ValueError("all overlap directions were discarded")
    values = eigenvalues[retained]
    transformation = (
        eigenvectors[:, retained]
        / torch.sqrt(values)[None, :]
        / term_norms[:, None]
    )
    return OverlapWhitening(
        transformation=transformation,
        eigenvalues=eigenvalues,
        raw_eigenvalues=raw_eigenvalues,
        term_norms=term_norms,
        retained_rank=rank,
        discarded_rank=overlap.shape[0] - rank,
    )


def contribution_gram_spectrum(
    overlap: torch.Tensor, amplitudes: torch.Tensor
) -> torch.Tensor:
    """Return the normalized spectrum of amplitude-weighted term contributions.

    This diagnostic is invariant under independent AGP scale/phase gauges and
    term permutations. It is not a particle entanglement spectrum.
    """

    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("overlap must be square")
    if amplitudes.shape != (overlap.shape[0],):
        raise ValueError("amplitudes must have shape (K,)")
    contribution_gram = (
        amplitudes.conj()[:, None] * overlap * amplitudes[None, :]
    )
    contribution_gram = 0.5 * (contribution_gram + contribution_gram.mH)
    normalization = torch.trace(contribution_gram).real
    if normalization <= 0:
        raise ValueError("state norm must be positive")
    spectrum = torch.linalg.eigvalsh(contribution_gram / normalization).real
    tolerance = 100 * torch.finfo(spectrum.dtype).eps
    if spectrum[0] < -tolerance:
        raise ValueError("contribution Gram matrix has a material negative eigenvalue")
    return torch.flip(torch.clamp(spectrum, min=0), dims=(0,))


def leave_one_out_energies(
    hamiltonian: torch.Tensor,
    overlap: torch.Tensor,
    *,
    relative_threshold: float = 1e-10,
) -> torch.Tensor:
    """Return optimized energies after removing each nonlinear AGP term."""

    if hamiltonian.shape != overlap.shape or overlap.ndim != 2:
        raise ValueError("hamiltonian and overlap must be equal square matrices")
    if overlap.shape[0] < 2:
        raise ValueError("leave-one-out diagnostics require at least two terms")
    energies = []
    for omitted in range(overlap.shape[0]):
        retained = torch.arange(overlap.shape[0], device=overlap.device) != omitted
        energies.append(
            solve_generalized_hermitian(
                hamiltonian[retained][:, retained],
                overlap[retained][:, retained],
                relative_threshold=relative_threshold,
            ).energy
        )
    return torch.stack(energies)


def assess_term_pruning(
    hamiltonian: torch.Tensor,
    overlap: torch.Tensor,
    *,
    condition_threshold: float = 1e8,
    energy_tolerance: float = 1e-8,
    relative_threshold: float = 1e-10,
) -> TermPruningAssessment:
    """Select a safely dispensable term only when the span is near-dependent.

    A large *raw* overlap condition never triggers pruning.  The rule requires
    either rank loss or a large gauge-balanced condition number, then chooses
    the term with the smallest fully reoptimized leave-one-out energy penalty.
    The returned penalty makes every proposed nonlinear deletion auditable.
    """

    if condition_threshold < 1:
        raise ValueError("condition_threshold must be at least one")
    if energy_tolerance < 0:
        raise ValueError("energy_tolerance must be nonnegative")
    solved = solve_generalized_hermitian(
        hamiltonian, overlap, relative_threshold=relative_threshold
    )
    near_dependent = (
        solved.discarded_rank > 0
        or solved.retained_condition_number >= condition_threshold
    )
    if not near_dependent:
        return TermPruningAssessment(
            should_prune=False,
            candidate=None,
            energy_penalty=None,
            balanced_condition_number=solved.retained_condition_number,
            discarded_rank=solved.discarded_rank,
            reason="balanced overlap is well conditioned",
        )
    leave_out = leave_one_out_energies(
        hamiltonian, overlap, relative_threshold=relative_threshold
    )
    penalties = leave_out - solved.energy
    candidate = int(torch.argmin(penalties).detach().cpu())
    penalty = float(penalties[candidate].detach().cpu())
    should_prune = penalty <= energy_tolerance
    return TermPruningAssessment(
        should_prune=should_prune,
        candidate=candidate,
        energy_penalty=penalty,
        balanced_condition_number=solved.retained_condition_number,
        discarded_rank=solved.discarded_rank,
        reason=(
            "near-dependent term is dispensable within the energy tolerance"
            if should_prune
            else "near-dependence detected but no term is safely dispensable"
        ),
    )


def solve_generalized_hermitian(
    hamiltonian: torch.Tensor,
    overlap: torch.Tensor,
    *,
    relative_threshold: float = 1e-10,
    absolute_threshold: float = 0.0,
) -> GeneralizedEigenResult:
    """Solve ``H c = E S c`` after removing ill-conditioned overlap modes."""

    if (
        hamiltonian.ndim != 2
        or hamiltonian.shape[0] != hamiltonian.shape[1]
        or overlap.shape != hamiltonian.shape
    ):
        raise ValueError("hamiltonian and overlap must be equal square matrices")
    if hamiltonian.dtype != overlap.dtype or hamiltonian.device != overlap.device:
        raise ValueError("hamiltonian and overlap must share dtype/device")
    if relative_threshold < 0 or absolute_threshold < 0:
        raise ValueError("overlap thresholds must be nonnegative")
    hermitian_overlap = 0.5 * (overlap + overlap.conj().transpose(0, 1))
    hermitian_hamiltonian = 0.5 * (
        hamiltonian + hamiltonian.conj().transpose(0, 1)
    )
    whitening = overlap_whitening(
        hermitian_overlap,
        relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
    )
    overlap_eigenvalues = whitening.eigenvalues
    raw_overlap_eigenvalues = whitening.raw_eigenvalues
    retained_rank = whitening.retained_rank
    retained_values = overlap_eigenvalues[-retained_rank:]
    whitener = whitening.transformation
    effective_hamiltonian = (
        whitener.conj().transpose(0, 1)
        @ hermitian_hamiltonian
        @ whitener
    )
    energies, effective_vectors = torch.linalg.eigh(effective_hamiltonian)
    energy = energies[0]
    amplitudes = whitener @ effective_vectors[:, 0]
    normalization = torch.einsum(
        "a,ab,b->", amplitudes.conj(), hermitian_overlap, amplitudes
    ).real
    amplitudes = amplitudes / torch.sqrt(normalization)
    residual = hermitian_hamiltonian @ amplitudes - energy * (
        hermitian_overlap @ amplitudes
    )
    raw_condition = (
        float((raw_overlap_eigenvalues[-1] / raw_overlap_eigenvalues[0]).detach().cpu())
        if raw_overlap_eigenvalues[0] > 0
        else None
    )
    return GeneralizedEigenResult(
        energy=energy,
        amplitudes=amplitudes,
        overlap_eigenvalues=overlap_eigenvalues,
        raw_overlap_eigenvalues=raw_overlap_eigenvalues,
        retained_rank=retained_rank,
        discarded_rank=whitening.discarded_rank,
        retained_condition_number=float(
            (retained_values[-1] / retained_values[0]).detach().cpu()
        ),
        raw_overlap_condition_number=raw_condition,
        residual_norm=torch.linalg.vector_norm(residual),
    )
