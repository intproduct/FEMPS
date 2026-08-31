"""Continuous ordered coordinates: center of mass and positive distances."""

from __future__ import annotations

import math

import torch


def _validate_particles(particles: int) -> None:
    if particles < 1:
        raise ValueError("particles must be positive")


def center_of_mass_distance_matrices(
    particles: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return matrices ``q=B x`` and ``x=A q`` for ordered coordinates.

    ``q[0]`` is the center of mass and ``q[i]=x[i]-x[i-1]`` for ``i>=1``.
    The two matrices have determinant magnitude one and are exact inverses up
    to the requested floating-point dtype.
    """

    _validate_particles(particles)
    forward = torch.zeros(
        particles, particles, dtype=dtype, device=device
    )
    forward[0] = 1 / particles
    if particles > 1:
        index = torch.arange(1, particles, device=device)
        forward[index, index] = 1
        forward[index, index - 1] = -1

    inverse = torch.zeros_like(forward)
    inverse[:, 0] = 1
    for coordinate in range(particles):
        for distance in range(1, particles):
            inverse[coordinate, distance] = (
                distance / particles
                if distance <= coordinate
                else -(particles - distance) / particles
            )
    return forward, inverse


def ordered_to_center_of_mass_distances(
    coordinates: torch.Tensor,
    *,
    validate_ordering: bool = True,
) -> torch.Tensor:
    """Map ordered coordinates to center of mass and positive distances."""

    if coordinates.ndim < 1 or coordinates.shape[-1] < 1:
        raise ValueError("coordinates must have a nonempty final particle axis")
    if validate_ordering and coordinates.shape[-1] > 1:
        if bool(torch.any(coordinates[..., 1:] <= coordinates[..., :-1])):
            raise ValueError("coordinates must be strictly ordered")
    forward, _ = center_of_mass_distance_matrices(
        coordinates.shape[-1],
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    return torch.einsum("ij,...j->...i", forward, coordinates)


def center_of_mass_distances_to_ordered(
    transformed: torch.Tensor,
    *,
    validate_distances: bool = True,
) -> torch.Tensor:
    """Invert :func:`ordered_to_center_of_mass_distances`."""

    if transformed.ndim < 1 or transformed.shape[-1] < 1:
        raise ValueError("transformed coordinates need a nonempty final axis")
    if validate_distances and transformed.shape[-1] > 1:
        if bool(torch.any(transformed[..., 1:] <= 0)):
            raise ValueError("interparticle distances must be positive")
    _, inverse = center_of_mass_distance_matrices(
        transformed.shape[-1],
        dtype=transformed.dtype,
        device=transformed.device,
    )
    return torch.einsum("ij,...j->...i", inverse, transformed)


def ordered_kinetic_metric(
    particles: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``G=B B^T`` in ``-1/2 sum_ab G_ab d_a d_b``."""

    forward, _ = center_of_mass_distance_matrices(
        particles, dtype=dtype, device=device
    )
    return forward @ forward.mT


def ordered_harmonic_metric(
    particles: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``K=A^T A`` in ``sum_i x_i^2/2 = q^T K q/2``."""

    _, inverse = center_of_mass_distance_matrices(
        particles, dtype=dtype, device=device
    )
    return inverse.mT @ inverse


def ordered_chamber_rescaling(particles: int) -> float:
    """Return ``sqrt(N!)`` relating full and unit-norm ordered wavefunctions."""

    _validate_particles(particles)
    return math.sqrt(math.factorial(particles))
