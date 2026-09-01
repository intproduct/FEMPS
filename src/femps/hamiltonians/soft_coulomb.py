"""Soft-Coulomb interactions in a harmonic-oscillator functional basis."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .harmonic_fermions import (
    FactorizedTwoBodyOperator,
    factorize_dense_two_body_operator,
)


@dataclass(frozen=True, slots=True)
class SoftCoulombDiagnostics:
    """Accuracy diagnostics for the quadrature-kernel factorization."""

    basis_order: int
    quadrature_order: int
    coupling: float
    softening: float
    retained_rank: int
    discarded_rank: int
    relative_threshold: float
    largest_kernel_eigenvalue_magnitude: float
    discarded_kernel_eigenvalue_magnitude: float
    dense_relative_factorization_error: float
    dense_hermiticity_residual: float
    exchange_symmetry_residual: float
    factorization_backend: str


def _validate_parameters(
    basis_order: int,
    quadrature_order: int,
    coupling: float,
    softening: float,
    relative_threshold: float,
) -> None:
    if basis_order < 1 or quadrature_order < 1:
        raise ValueError("basis_order and quadrature_order must be positive")
    if not math.isfinite(coupling) or coupling < 0:
        raise ValueError("coupling must be finite and nonnegative")
    if not math.isfinite(softening) or softening <= 0:
        raise ValueError("softening must be finite and positive")
    if not 0 <= relative_threshold < 1:
        raise ValueError("relative_threshold must lie in [0,1)")


def _gauss_hermite_rule(
    order: int,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return (
        torch.as_tensor(nodes, dtype=torch.float64, device=device),
        torch.as_tensor(weights, dtype=torch.float64, device=device),
    )


def _normalized_hermite_polynomials(
    basis_order: int, nodes: torch.Tensor
) -> torch.Tensor:
    """Return ``H_n(x)/sqrt(2^n n! sqrt(pi))`` by stable recurrence."""

    values = torch.empty(
        (basis_order, nodes.numel()), dtype=nodes.dtype, device=nodes.device
    )
    values[0] = math.pi ** (-0.25)
    if basis_order == 1:
        return values
    values[1] = math.sqrt(2.0) * nodes * values[0]
    for degree in range(1, basis_order - 1):
        values[degree + 1] = (
            math.sqrt(2.0 / (degree + 1)) * nodes * values[degree]
            - math.sqrt(degree / (degree + 1)) * values[degree - 1]
        )
    return values


def _quadrature_components(
    basis_order: int,
    quadrature_order: int,
    softening: float,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = _gauss_hermite_rule(quadrature_order, device=device)
    polynomials = _normalized_hermite_polynomials(basis_order, nodes)
    densities = torch.einsum("pi,ri->ipr", polynomials, polynomials)
    weighted_densities = torch.sqrt(weights)[:, None, None] * densities
    separation = nodes[:, None] - nodes[None, :]
    kernel = torch.rsqrt(separation.square() + softening * softening)
    weighted_kernel = torch.sqrt(weights)[:, None] * kernel * torch.sqrt(weights)[None, :]
    return weighted_densities, weighted_kernel


def soft_coulomb_dense_quadrature(
    basis_order: int,
    *,
    quadrature_order: int,
    coupling: float = 1.0,
    softening: float = 1.0,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Direct four-index Gauss-Hermite tensor, without kernel factorization.

    The convention is ``V[p,q,r,s] = <p q|g/sqrt((x-y)^2+a^2)|r s>``.
    """

    _validate_parameters(
        basis_order, quadrature_order, coupling, softening, 0.0
    )
    densities, kernel = _quadrature_components(
        basis_order, quadrature_order, softening, device=device
    )
    dense = coupling * torch.einsum(
        "ipr,ij,jqs->pqrs", densities, kernel, densities
    )
    return dense.to(dtype=dtype)


def soft_coulomb_operator(
    basis_order: int,
    *,
    quadrature_order: int,
    coupling: float = 1.0,
    softening: float = 1.0,
    relative_threshold: float = 1e-12,
    factorization_backend: str = "kernel",
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str = "cpu",
) -> tuple[FactorizedTwoBodyOperator, SoftCoulombDiagnostics]:
    """Build a factorized soft-Coulomb pair operator and its diagnostics."""

    _validate_parameters(
        basis_order,
        quadrature_order,
        coupling,
        softening,
        relative_threshold,
    )
    densities, kernel = _quadrature_components(
        basis_order, quadrature_order, softening, device=device
    )
    direct = coupling * torch.einsum(
        "ipr,ij,jqs->pqrs", densities, kernel, densities
    ).to(dtype)
    direct_norm = torch.linalg.vector_norm(direct)
    matrix = direct.reshape(basis_order**2, basis_order**2)
    hermiticity = torch.linalg.vector_norm(matrix - matrix.mH) / direct_norm
    exchange = torch.linalg.vector_norm(direct - direct.permute(1, 0, 3, 2)) / direct_norm
    if factorization_backend == "physical":
        operator, physical = factorize_dense_two_body_operator(
            direct, relative_threshold=relative_threshold
        )
        return operator, SoftCoulombDiagnostics(
            basis_order=basis_order,
            quadrature_order=quadrature_order,
            coupling=coupling,
            softening=softening,
            retained_rank=physical.retained_rank,
            discarded_rank=physical.discarded_rank,
            relative_threshold=relative_threshold,
            largest_kernel_eigenvalue_magnitude=0.0,
            discarded_kernel_eigenvalue_magnitude=0.0,
            dense_relative_factorization_error=physical.dense_relative_error,
            dense_hermiticity_residual=float(hermiticity.detach().cpu()),
            exchange_symmetry_residual=float(exchange.detach().cpu()),
            factorization_backend="physical_operator_svd",
        )
    if factorization_backend != "kernel":
        raise ValueError("factorization_backend must be kernel or physical")
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    largest = torch.max(torch.abs(eigenvalues))
    keep = torch.abs(eigenvalues) > relative_threshold * largest
    retained_values = eigenvalues[keep]
    retained_vectors = eigenvectors[:, keep]
    factors = torch.einsum("ipr,il->lpr", densities, retained_vectors).to(dtype)
    weights = (coupling * retained_values).to(dtype)
    operator = FactorizedTwoBodyOperator(factors, factors, weights)

    reconstructed = operator.dense()
    factorization_error = torch.linalg.vector_norm(reconstructed - direct) / direct_norm
    discarded = torch.abs(eigenvalues[~keep])
    diagnostics = SoftCoulombDiagnostics(
        basis_order=basis_order,
        quadrature_order=quadrature_order,
        coupling=coupling,
        softening=softening,
        retained_rank=int(torch.count_nonzero(keep)),
        discarded_rank=int(torch.count_nonzero(~keep)),
        relative_threshold=relative_threshold,
        largest_kernel_eigenvalue_magnitude=float(largest.detach().cpu()),
        discarded_kernel_eigenvalue_magnitude=(
            float(torch.max(discarded).detach().cpu()) if discarded.numel() else 0.0
        ),
        dense_relative_factorization_error=float(factorization_error.detach().cpu()),
        dense_hermiticity_residual=float(hermiticity.detach().cpu()),
        exchange_symmetry_residual=float(exchange.detach().cpu()),
        factorization_backend="quadrature_kernel_eigh",
    )
    return operator, diagnostics


def soft_coulomb_two_fermion_relative_grid_energy(
    *,
    intervals: int,
    half_width: float = 8.0,
    coupling: float = 1.0,
    softening: float = 1.0,
) -> float:
    """Independent N=2 energy from the odd relative-coordinate half-line.

    With ``R=(x1+x2)/sqrt(2)`` and ``r=(x1-x2)/sqrt(2)``, the center-of-mass
    ground energy is ``1/2``. Antisymmetry imposes Dirichlet values at ``r=0``;
    a second Dirichlet boundary is placed at ``r=half_width``.
    """

    _validate_parameters(1, intervals, coupling, softening, 0.0)
    if intervals < 3 or not math.isfinite(half_width) or half_width <= 0:
        raise ValueError("intervals >= 3 and finite half_width > 0 are required")
    spacing = half_width / intervals
    coordinate = spacing * torch.arange(1, intervals, dtype=torch.float64)
    potential = (
        0.5 * coordinate.square()
        + coupling
        * torch.rsqrt(2.0 * coordinate.square() + softening * softening)
    )
    diagonal = 1.0 / spacing**2 + potential
    off_diagonal = torch.full(
        (intervals - 2,), -0.5 / spacing**2, dtype=torch.float64
    )
    relative_hamiltonian = torch.diag(diagonal)
    relative_hamiltonian += torch.diag(off_diagonal, diagonal=1)
    relative_hamiltonian += torch.diag(off_diagonal, diagonal=-1)
    relative_energy = torch.linalg.eigvalsh(relative_hamiltonian)[0]
    return 0.5 + float(relative_energy)
