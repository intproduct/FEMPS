"""Exponential-cost reference operations for small antisymmetric tensors.

The functions in this module intentionally materialize tensors in ``V**N``.
They are validation oracles, not scalable solver primitives.
"""

from __future__ import annotations

import itertools
import math

import torch


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _validate_particle_tensor(tensor: torch.Tensor) -> None:
    if tensor.ndim < 1:
        raise ValueError("a particle tensor must have at least one axis")
    if len(set(tensor.shape)) != 1:
        raise ValueError("all particle axes must use the same one-particle dimension")


def _validate_orbitals(orbitals: torch.Tensor) -> tuple[int, int]:
    if orbitals.ndim != 2:
        raise ValueError("orbitals must have shape (one_particle_dimension, particles)")
    dimension, particles = orbitals.shape
    if particles < 1:
        raise ValueError("at least one orbital is required")
    if dimension < particles:
        raise ValueError("one-particle dimension must be at least the particle count")
    return dimension, particles


def alternating_projection(tensor: torch.Tensor) -> torch.Tensor:
    """Return the orthogonal projector of ``tensor`` onto ``Lambda^N V``."""

    _validate_particle_tensor(tensor)
    particles = tensor.ndim
    projected = torch.zeros_like(tensor)
    for permutation in itertools.permutations(range(particles)):
        projected = projected + _permutation_sign(permutation) * tensor.permute(permutation)
    return projected / math.factorial(particles)


def normalized_slater_from_antisymmetrizer(orbitals: torch.Tensor) -> torch.Tensor:
    """Materialize ``u_1 wedge ... wedge u_N`` by antisymmetrization.

    The convention is ``1/sqrt(N!)`` times the signed permutation sum. Its
    squared norm is the determinant of the orbital Gram matrix, and is one for
    orthonormal columns.
    """

    _, particles = _validate_orbitals(orbitals)
    product = orbitals[:, 0]
    for column in range(1, particles):
        product = torch.tensordot(product, orbitals[:, column], dims=0)
    return math.sqrt(math.factorial(particles)) * alternating_projection(product)


def normalized_slater_from_minors(orbitals: torch.Tensor) -> torch.Tensor:
    """Independently materialize a normalized-convention wedge using minors."""

    dimension, particles = _validate_orbitals(orbitals)
    result = torch.zeros(
        (dimension,) * particles,
        dtype=orbitals.dtype,
        device=orbitals.device,
    )
    scale = math.sqrt(math.factorial(particles))
    permutations = list(itertools.permutations(range(particles)))
    for support in itertools.combinations(range(dimension), particles):
        minor = torch.linalg.det(orbitals[list(support), :]) / scale
        for permutation in permutations:
            index = tuple(support[position] for position in permutation)
            result[index] = _permutation_sign(permutation) * minor
    return result


def exterior_coefficients_to_tensor(
    coefficients: torch.Tensor, dimension: int, particles: int
) -> torch.Tensor:
    """Materialize increasing-basis exterior coefficients as a particle tensor.

    The increasing Slater basis is orthonormal in the normalized tensor
    convention.  This routine is an exponential small-system oracle.
    """

    if coefficients.ndim != 1:
        raise ValueError("coefficients must be a vector")
    if particles < 1 or dimension < particles:
        raise ValueError("require dimension >= particles >= 1")
    supports = list(itertools.combinations(range(dimension), particles))
    if coefficients.numel() != len(supports):
        raise ValueError("coefficient count must equal binom(dimension, particles)")
    result = torch.zeros(
        (dimension,) * particles,
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    scale = math.sqrt(math.factorial(particles))
    permutations = list(itertools.permutations(range(particles)))
    for support, coefficient in zip(supports, coefficients, strict=True):
        for permutation in permutations:
            index = tuple(support[position] for position in permutation)
            result[index] = _permutation_sign(permutation) * coefficient / scale
    return result


def one_body_expectation_exterior_coefficients(
    coefficients: torch.Tensor,
    operator: torch.Tensor,
    particles: int,
) -> torch.Tensor:
    """Evaluate ``<c|dGamma(operator)|c>`` in the increasing exterior basis."""

    if coefficients.ndim != 1 or operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("require a coefficient vector and a square one-body operator")
    dimension = operator.shape[0]
    supports = list(itertools.combinations(range(dimension), particles))
    if coefficients.numel() != len(supports):
        raise ValueError("coefficient count must equal binom(dimension, particles)")
    lookup = {support: position for position, support in enumerate(supports)}
    acted = torch.zeros_like(coefficients)
    for ket_position, support in enumerate(supports):
        ket_coefficient = coefficients[ket_position]
        for annihilation_position, removed in enumerate(support):
            remaining = support[:annihilation_position] + support[annihilation_position + 1 :]
            annihilation_sign = -1 if annihilation_position % 2 else 1
            for created in range(dimension):
                if created in remaining:
                    continue
                creation_position = sum(index < created for index in remaining)
                creation_sign = -1 if creation_position % 2 else 1
                target = tuple(sorted(remaining + (created,)))
                acted[lookup[target]] = acted[lookup[target]] + (
                    annihilation_sign
                    * creation_sign
                    * operator[created, removed]
                    * ket_coefficient
                )
    return torch.vdot(coefficients, acted)


def one_body_density_exterior_coefficients(
    coefficients: torch.Tensor,
    dimension: int,
    particles: int,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Return ``gamma[p,q] = <a_q^dagger a_p>`` in the exterior basis."""

    if coefficients.ndim != 1 or particles < 1 or dimension < particles:
        raise ValueError("require a vector and dimension >= particles >= 1")
    supports = list(itertools.combinations(range(dimension), particles))
    if coefficients.numel() != len(supports):
        raise ValueError("coefficient count must equal binom(dimension, particles)")
    lookup = {support: position for position, support in enumerate(supports)}
    density = torch.zeros(
        dimension, dimension, dtype=coefficients.dtype, device=coefficients.device
    )
    for ket_position, support in enumerate(supports):
        for annihilation_position, removed in enumerate(support):
            remaining = support[:annihilation_position] + support[annihilation_position + 1 :]
            annihilation_sign = -1 if annihilation_position % 2 else 1
            for created in range(dimension):
                if created in remaining:
                    continue
                creation_position = sum(index < created for index in remaining)
                creation_sign = -1 if creation_position % 2 else 1
                target = tuple(sorted(remaining + (created,)))
                density[removed, created] = density[removed, created] + (
                    annihilation_sign
                    * creation_sign
                    * coefficients[lookup[target]].conj()
                    * coefficients[ket_position]
                )
    if normalize:
        norm = torch.vdot(coefficients, coefficients).real
        if norm <= 0:
            raise ValueError("cannot normalize a zero exterior state")
        density = density / norm
    return density


def fermionic_correlation_multiplicity(
    coefficients: torch.Tensor,
    dimension: int,
    particles: int,
) -> torch.Tensor:
    """Return ``N / Tr(gamma^2)``, equal to one for a single Slater.

    This gauge-invariant one-body purity diagnostic is called a correlation
    multiplicity here.  It is not labeled particle entanglement.
    """

    density = one_body_density_exterior_coefficients(
        coefficients, dimension, particles, normalize=True
    )
    purity = torch.trace(density @ density).real
    return particles / purity


def antisymmetry_residual(tensor: torch.Tensor, *, relative: bool = True) -> torch.Tensor:
    """Return the largest adjacent-transposition residual in Frobenius norm."""

    _validate_particle_tensor(tensor)
    if tensor.ndim == 1:
        return torch.zeros((), dtype=tensor.real.dtype, device=tensor.device)
    residuals = []
    for axis in range(tensor.ndim - 1):
        permutation = list(range(tensor.ndim))
        permutation[axis], permutation[axis + 1] = permutation[axis + 1], permutation[axis]
        residuals.append(torch.linalg.vector_norm(tensor + tensor.permute(permutation)))
    residual = torch.stack(residuals).max()
    if not relative:
        return residual
    norm = torch.linalg.vector_norm(tensor)
    if norm == 0:
        return residual
    return residual / norm


def particle_unfolding(tensor: torch.Tensor, cut: int) -> torch.Tensor:
    """Return the ordinary particle ``cut | N-cut`` matricization."""

    _validate_particle_tensor(tensor)
    particles = tensor.ndim
    if not 1 <= cut < particles:
        raise ValueError("cut must satisfy 1 <= cut < particle count")
    dimension = tensor.shape[0]
    return tensor.reshape(dimension**cut, dimension ** (particles - cut))


def particle_schmidt_spectrum(tensor: torch.Tensor, cut: int) -> torch.Tensor:
    """Return descending singular values of a particle unfolding."""

    return torch.linalg.svdvals(particle_unfolding(tensor, cut))


def _numerical_rank(singular_values: torch.Tensor, rows: int, columns: int) -> int:
    if singular_values.numel() == 0 or singular_values[0] == 0:
        return 0
    tolerance = max(rows, columns) * torch.finfo(singular_values.dtype).eps * singular_values[0]
    return int(torch.count_nonzero(singular_values > tolerance).item())


def particle_tt_ranks(tensor: torch.Tensor) -> tuple[int, ...]:
    """Return exact-up-to-roundoff minimal ordinary TT bond ranks."""

    _validate_particle_tensor(tensor)
    ranks = []
    for cut in range(1, tensor.ndim):
        unfolding = particle_unfolding(tensor, cut)
        singular_values = torch.linalg.svdvals(unfolding)
        ranks.append(_numerical_rank(singular_values, *unfolding.shape))
    return tuple(ranks)


def best_rank_error(
    singular_values: torch.Tensor,
    rank: int,
    *,
    relative: bool = True,
) -> torch.Tensor:
    """Return the Eckart--Young Frobenius error after rank-``rank`` truncation."""

    if singular_values.ndim != 1:
        raise ValueError("singular_values must be one-dimensional")
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    tail = singular_values[min(rank, singular_values.numel()) :]
    error = torch.linalg.vector_norm(tail)
    if not relative:
        return error
    total = torch.linalg.vector_norm(singular_values)
    if total == 0:
        return error
    return error / total


def slater_flat_spectrum(
    particles: int,
    cut: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the theorem-predicted spectrum of a normalized Slater state."""

    if particles < 2:
        raise ValueError("particles must be at least two")
    if not 1 <= cut < particles:
        raise ValueError("cut must satisfy 1 <= cut < particles")
    multiplicity = math.comb(particles, cut)
    return torch.full(
        (multiplicity,),
        1.0 / math.sqrt(multiplicity),
        dtype=dtype,
        device=device,
    )
