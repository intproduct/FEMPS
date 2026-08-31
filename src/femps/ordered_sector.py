"""Exact small-system oracles for an ordered-coordinate Weyl chamber."""

from __future__ import annotations

import itertools
import math

import torch


def ordered_configurations(local_dimension: int, particles: int) -> tuple[tuple[int, ...], ...]:
    """Return strictly increasing coordinate configurations."""

    if particles < 1 or local_dimension < particles:
        raise ValueError("require 1 <= particles <= local_dimension")
    return tuple(itertools.combinations(range(local_dimension), particles))


def restrict_to_ordered_sector(state: torch.Tensor) -> torch.Tensor:
    """Apply the normalized restriction ``sqrt(N!) Psi|_{x1<...<xN}``.

    The input is an explicit antisymmetric particle tensor. This routine is an
    exponential truth oracle and is not a production representation.
    """

    particles = state.ndim
    if particles < 1 or len(set(state.shape)) != 1:
        raise ValueError("state must have N equal coordinate dimensions")
    supports = ordered_configurations(state.shape[0], particles)
    values = torch.stack([state[support] for support in supports])
    return math.sqrt(math.factorial(particles)) * values


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def extend_from_ordered_sector(
    values: torch.Tensor,
    local_dimension: int,
    particles: int,
) -> torch.Tensor:
    """Invert the normalized ordered-sector restriction by antisymmetry."""

    supports = ordered_configurations(local_dimension, particles)
    if values.ndim != 1 or values.numel() != len(supports):
        raise ValueError("values must contain one amplitude per ordered configuration")
    state = torch.zeros(
        (local_dimension,) * particles,
        dtype=values.dtype,
        device=values.device,
    )
    normalization = math.sqrt(math.factorial(particles))
    for support, value in zip(supports, values, strict=True):
        for permutation in itertools.permutations(range(particles)):
            coordinate = tuple(support[index] for index in permutation)
            state[coordinate] = _permutation_sign(permutation) * value / normalization
    return state


def ordered_sector_hamiltonian(
    one_body: torch.Tensor,
    particles: int,
    *,
    pair_potential: torch.Tensor | None = None,
    locality_tolerance: float = 1e-12,
) -> torch.Tensor:
    """Project a local coordinate-grid Hamiltonian onto ``x1<...<xN``.

    ``one_body`` must be tridiagonal in the coordinate grid. This locality is
    essential: a nonlocal hop can jump across another labeled particle and is
    not the hard-wall discretization of a differential kinetic operator.
    ``pair_potential[i,j]`` is an optional diagonal coordinate-space pair term.
    """

    if one_body.ndim != 2 or one_body.shape[0] != one_body.shape[1]:
        raise ValueError("one_body must be square")
    dimension = one_body.shape[0]
    if not torch.allclose(one_body, one_body.mH, atol=locality_tolerance, rtol=0):
        raise ValueError("one_body must be Hermitian")
    row, column = torch.meshgrid(
        torch.arange(dimension, device=one_body.device),
        torch.arange(dimension, device=one_body.device),
        indexing="ij",
    )
    nonlocal_entries = torch.abs(one_body) * ((row - column).abs() > 1)
    if torch.any(nonlocal_entries > locality_tolerance):
        raise ValueError("one_body must be tridiagonal in coordinate order")
    if pair_potential is not None:
        if pair_potential.shape != one_body.shape:
            raise ValueError("pair_potential must have shape (D,D)")
        if not torch.allclose(
            pair_potential, pair_potential.mH, atol=locality_tolerance, rtol=0
        ):
            raise ValueError("pair_potential must be Hermitian")

    supports = ordered_configurations(dimension, particles)
    support_index = {support: index for index, support in enumerate(supports)}
    hamiltonian = torch.zeros(
        (len(supports), len(supports)),
        dtype=one_body.dtype,
        device=one_body.device,
    )
    for column_index, support in enumerate(supports):
        for particle, occupied in enumerate(support):
            for target in range(max(0, occupied - 1), min(dimension, occupied + 2)):
                candidate = list(support)
                candidate[particle] = target
                if any(candidate[index] >= candidate[index + 1] for index in range(particles - 1)):
                    continue
                row_index = support_index[tuple(candidate)]
                hamiltonian[row_index, column_index] += one_body[target, occupied]
        if pair_potential is not None:
            hamiltonian[column_index, column_index] += sum(
                pair_potential[support[left], support[right]]
                for left in range(particles)
                for right in range(left + 1, particles)
            )
    return hamiltonian


def finite_difference_harmonic_hamiltonian(
    grid_points: int,
    spacing: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a centered real-space grid and open-boundary HO Hamiltonian."""

    if grid_points < 2 or spacing <= 0:
        raise ValueError("grid_points >= 2 and spacing > 0 are required")
    grid = spacing * (
        torch.arange(grid_points, dtype=torch.float64, device=device)
        - (grid_points - 1) / 2
    )
    hamiltonian = torch.diag(1 / spacing**2 + 0.5 * grid**2).to(dtype)
    hopping = torch.full(
        (grid_points - 1,),
        -0.5 / spacing**2,
        dtype=dtype,
        device=device,
    )
    hamiltonian += torch.diag(hopping, diagonal=1)
    hamiltonian += torch.diag(hopping.conj(), diagonal=-1)
    return grid, hamiltonian
