"""Conditioned linear-amplitude solve for a fixed finite-AGP basis."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GeneralizedEigenResult:
    """Lowest generalized Hermitian eigenpair and overlap diagnostics."""

    energy: torch.Tensor
    amplitudes: torch.Tensor
    overlap_eigenvalues: torch.Tensor
    retained_rank: int
    discarded_rank: int
    retained_condition_number: float
    residual_norm: torch.Tensor


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
    overlap_eigenvalues, overlap_eigenvectors = torch.linalg.eigh(
        hermitian_overlap
    )
    largest = overlap_eigenvalues[-1]
    threshold = torch.maximum(
        torch.as_tensor(
            absolute_threshold,
            dtype=overlap_eigenvalues.dtype,
            device=overlap.device,
        ),
        relative_threshold * largest,
    )
    if largest <= 0:
        raise ValueError("overlap matrix has no positive direction")
    if overlap_eigenvalues[0] < -threshold:
        raise ValueError("overlap matrix has a material negative eigenvalue")
    retained = overlap_eigenvalues > threshold
    retained_rank = int(torch.count_nonzero(retained).item())
    if retained_rank == 0:
        raise ValueError("all overlap directions were discarded")
    retained_values = overlap_eigenvalues[retained]
    retained_vectors = overlap_eigenvectors[:, retained]
    whitener = retained_vectors / torch.sqrt(retained_values)[None, :]
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
    return GeneralizedEigenResult(
        energy=energy,
        amplitudes=amplitudes,
        overlap_eigenvalues=overlap_eigenvalues,
        retained_rank=retained_rank,
        discarded_rank=overlap.shape[0] - retained_rank,
        retained_condition_number=float(
            (retained_values[-1] / retained_values[0]).detach().cpu()
        ),
        residual_norm=torch.linalg.vector_norm(residual),
    )
