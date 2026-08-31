"""Controlled interval-polynomial interactions for continuous ordered MPS."""

from __future__ import annotations

import math

import numpy as np
import torch

from femps.basis.dirichlet_sine import (
    dirichlet_sine_affine_power_matrices,
)
from femps.baselines.ordered_distance_mpo import sum_mpos


def soft_coulomb_chebyshev_power_coefficients(
    maximum_separation: float,
    degree: int,
    *,
    softening: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Approximate soft Coulomb as powers of ``t=2s/S-1`` on ``0<=s<=S``."""

    if maximum_separation <= 0 or degree < 0 or softening <= 0:
        raise ValueError("invalid separation, degree, or softening")
    from numpy.polynomial import Chebyshev, Polynomial

    approximation = Chebyshev.interpolate(
        lambda transformed: 1
        / np.sqrt(
            (maximum_separation * (transformed + 1) / 2) ** 2
            + softening**2
        ),
        degree,
    )
    polynomial = approximation.convert(kind=Polynomial)
    return torch.as_tensor(polynomial.coef, dtype=dtype, device=device)


def soft_coulomb_chebyshev_sampled_error(
    maximum_separation: float,
    degree: int,
    *,
    softening: float = 1.0,
    samples: int = 20001,
) -> float:
    """Return a dense sampled scalar error (diagnostic, not an interval proof)."""

    if samples < 2:
        raise ValueError("samples must be at least two")
    coefficients = soft_coulomb_chebyshev_power_coefficients(
        maximum_separation, degree, softening=softening
    ).numpy()
    separation = np.linspace(0, maximum_separation, samples)
    transformed = 2 * separation / maximum_separation - 1
    observed = np.polynomial.polynomial.polyval(transformed, coefficients)
    expected = 1 / np.sqrt(separation**2 + softening**2)
    return float(np.max(np.abs(observed - expected)))


def ordered_continuous_soft_coulomb_pair_mpo(
    particles: int,
    basis_order: int,
    distance_length: float,
    left_particle: int,
    right_particle: int,
    degree: int,
    *,
    softening: float = 1.0,
    quadrature_order: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Return a Chebyshev-polynomial MPO for one distance-interval pair.

    The first interval site represents ``2r/S-1`` and later sites represent
    ``2r/S``.  Their sum is exactly the Chebyshev coordinate
    ``t=2(x_j-x_i)/S-1``.  A degree-``K`` power automaton has bond ``K+1``.
    """

    if (
        particles < 2
        or basis_order < 1
        or distance_length <= 0
        or not 0 <= left_particle < right_particle < particles
        or degree < 0
        or softening <= 0
    ):
        raise ValueError("invalid particles, basis, interval, degree, or softening")
    try:
        from latticetn.mpo import MPO
    except ImportError as exc:
        raise ImportError("latticeTN is required for continuous interaction MPOs") from exc

    interval_sites = right_particle - left_particle
    maximum_separation = interval_sites * distance_length
    coefficients = soft_coulomb_chebyshev_power_coefficients(
        maximum_separation,
        degree,
        softening=softening,
        dtype=dtype,
        device=device,
    )
    first_powers = dirichlet_sine_affine_power_matrices(
        basis_order,
        distance_length,
        degree,
        scale=2 / maximum_separation,
        shift=-1,
        quadrature_order=quadrature_order,
        dtype=dtype,
        device=device,
    )
    later_powers = dirichlet_sine_affine_power_matrices(
        basis_order,
        distance_length,
        degree,
        scale=2 / maximum_separation,
        shift=0,
        quadrature_order=quadrature_order,
        dtype=dtype,
        device=device,
    )
    start = left_particle + 1
    end = right_particle
    bond = degree + 1
    identity = torch.eye(basis_order, dtype=dtype, device=device)
    tensors = []
    for site in range(particles):
        if site < start or site > end:
            tensors.append(
                identity.transpose(0, 1).reshape(
                    1, 1, basis_order, basis_order
                )
            )
            continue
        if start == end:
            operator = sum(
                coefficients[power] * first_powers[power]
                for power in range(bond)
            )
            tensors.append(
                operator.transpose(0, 1).reshape(
                    1, 1, basis_order, basis_order
                )
            )
            continue
        if site == start:
            tensor = torch.zeros(
                1,
                bond,
                basis_order,
                basis_order,
                dtype=dtype,
                device=device,
            )
            for power in range(bond):
                tensor[0, power] = first_powers[power].transpose(0, 1)
            tensors.append(tensor)
            continue
        if site == end:
            tensor = torch.zeros(
                bond,
                1,
                basis_order,
                basis_order,
                dtype=dtype,
                device=device,
            )
            for accumulated_power in range(bond):
                operator = torch.zeros_like(identity)
                for total_power in range(accumulated_power, bond):
                    operator = operator + (
                        math.comb(total_power, accumulated_power)
                        * coefficients[total_power]
                        * later_powers[total_power - accumulated_power]
                    )
                tensor[accumulated_power, 0] = operator.transpose(0, 1)
            tensors.append(tensor)
            continue
        tensor = torch.zeros(
            bond,
            bond,
            basis_order,
            basis_order,
            dtype=dtype,
            device=device,
        )
        for accumulated_power in range(bond):
            for total_power in range(accumulated_power, bond):
                tensor[accumulated_power, total_power] = (
                    math.comb(total_power, accumulated_power)
                    * later_powers[total_power - accumulated_power].transpose(0, 1)
                )
        tensors.append(tensor)
    return MPO(
        tensors,
        length=particles,
        dim=basis_order,
        dtype=dtype,
        device=device,
    )


def ordered_continuous_soft_coulomb_mpo(
    particles: int,
    basis_order: int,
    distance_length: float,
    degree: int,
    *,
    softening: float = 1.0,
    quadrature_order: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Return the block-direct-sum MPO of all soft-Coulomb particle pairs."""

    return sum_mpos(
        [
            ordered_continuous_soft_coulomb_pair_mpo(
                particles,
                basis_order,
                distance_length,
                left,
                right,
                degree,
                softening=softening,
                quadrature_order=quadrature_order,
                dtype=dtype,
                device=device,
            )
            for left in range(particles)
            for right in range(left + 1, particles)
        ]
    )
