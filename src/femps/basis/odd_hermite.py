"""Odd harmonic-oscillator functions restricted to the positive half-line."""

from __future__ import annotations

import math

import numpy as np
import torch


def _validate(order: int, length_scale: float) -> None:
    if order < 1:
        raise ValueError("basis order must be positive")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")


def _normalized_hermite_polynomials(
    maximum_quantum_number: int, points: torch.Tensor
) -> torch.Tensor:
    """Return ``H_n(x)/(pi^(1/4) sqrt(2^n n!))`` by stable recurrence."""

    values = torch.empty(
        maximum_quantum_number + 1,
        points.numel(),
        dtype=points.dtype,
        device=points.device,
    )
    values[0] = math.pi ** (-0.25)
    if maximum_quantum_number == 0:
        return values
    values[1] = math.sqrt(2) * points * values[0]
    for quantum_number in range(1, maximum_quantum_number):
        values[quantum_number + 1] = (
            math.sqrt(2 / (quantum_number + 1))
            * points
            * values[quantum_number]
            - math.sqrt(quantum_number / (quantum_number + 1))
            * values[quantum_number - 1]
        )
    return values


def odd_hermite_basis_values(
    order: int,
    points: torch.Tensor,
    length_scale: float = 1.0,
) -> torch.Tensor:
    """Evaluate normalized odd Hermite functions on ``r>0``."""

    _validate(order, length_scale)
    scaled = points / length_scale
    polynomials = _normalized_hermite_polynomials(2 * order - 1, scaled)
    odd = polynomials[1::2].mT
    gaussian = torch.exp(-0.5 * scaled.square())
    return math.sqrt(2 / length_scale) * odd * gaussian[..., None]


def _laguerre_data(
    order: int,
    quadrature_order: int,
    *,
    alpha: float = 0.0,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")
    if alpha <= -1:
        raise ValueError("generalized Laguerre alpha must exceed -1")
    index = np.arange(quadrature_order, dtype=np.float64)
    diagonal = 2 * index + 1 + alpha
    off_diagonal = np.sqrt(
        np.arange(1, quadrature_order, dtype=np.float64)
        * (np.arange(1, quadrature_order, dtype=np.float64) + alpha)
    )
    jacobi = np.diag(diagonal)
    jacobi += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    nodes_numpy, eigenvectors = np.linalg.eigh(jacobi)
    weights_numpy = math.gamma(alpha + 1) * np.square(eigenvectors[0])
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    nodes = torch.as_tensor(nodes_numpy, dtype=real_dtype, device=device)
    weights = torch.as_tensor(weights_numpy, dtype=real_dtype, device=device)
    roots = torch.sqrt(nodes)
    polynomials = _normalized_hermite_polynomials(2 * order - 1, roots).to(dtype)
    return nodes.to(dtype), weights.to(dtype), polynomials


def odd_hermite_power_matrices(
    order: int,
    maximum_power: int,
    length_scale: float = 1.0,
    *,
    quadrature_order: int = 256,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return half-line Galerkin matrices ``<m|r^p|n>``."""

    _validate(order, length_scale)
    if maximum_power < 0:
        raise ValueError("maximum_power must be nonnegative")
    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(quadrature_order)
    dimensionless_cutoff = math.sqrt(4 * order + 2) + 8
    physical_cutoff = length_scale * dimensionless_cutoff
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    nodes = torch.as_tensor(
        (raw_nodes + 1) * physical_cutoff / 2,
        dtype=real_dtype,
        device=device,
    )
    weights = torch.as_tensor(
        raw_weights * physical_cutoff / 2,
        dtype=real_dtype,
        device=device,
    )
    basis = odd_hermite_basis_values(order, nodes, length_scale).to(dtype)
    matrices = []
    for power in range(maximum_power + 1):
        matrices.append(
            torch.einsum(
                "xm,x,x,xn->mn",
                basis.conj(),
                weights,
                nodes.to(dtype) ** power,
                basis,
            )
        )
    return tuple(matrices)


def odd_hermite_derivative_matrix(
    order: int,
    length_scale: float = 1.0,
    *,
    quadrature_order: int = 96,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the half-line projection ``<m|d/dr|n>``."""

    _validate(order, length_scale)
    nodes, weights, polynomials = _laguerre_data(
        order,
        quadrature_order,
        dtype=dtype,
        device=device,
    )
    roots = torch.sqrt(nodes)
    odd_quantum_numbers = torch.arange(
        1, 2 * order, 2, dtype=nodes.real.dtype, device=nodes.device
    ).to(dtype)
    odd = polynomials[1::2]
    derivative_polynomial = (
        torch.sqrt(2 * odd_quantum_numbers)[:, None] * polynomials[0:-1:2]
        - roots[None, :] * odd
    )
    factor = weights / torch.sqrt(nodes) / length_scale
    return torch.einsum("mx,x,nx->mn", odd, factor, derivative_polynomial)


def odd_hermite_position_matrix(
    order: int,
    length_scale: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Return ``<m|r|n>`` on the positive half-line."""

    return odd_hermite_power_matrices(
        order, 1, length_scale, **kwargs
    )[1]


def odd_hermite_position_squared_matrix(
    order: int,
    length_scale: float = 1.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the analytic odd-sector projection ``<m|r^2|n>``."""

    _validate(order, length_scale)
    result = torch.zeros(order, order, dtype=dtype, device=device)
    quantum = torch.arange(1, 2 * order, 2, dtype=torch.float64, device=device)
    result.diagonal().copy_((length_scale**2 * (quantum + 0.5)).to(dtype))
    if order > 1:
        values = 0.5 * length_scale**2 * torch.sqrt(
            (quantum[:-1] + 1) * (quantum[:-1] + 2)
        )
        index = torch.arange(order - 1, device=device)
        result[index, index + 1] = values.to(dtype)
        result[index + 1, index] = values.to(dtype)
    return result


def odd_hermite_negative_second_derivative_matrix(
    order: int,
    length_scale: float = 1.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the analytic projection ``<m|-d^2/dr^2|n>``."""

    _validate(order, length_scale)
    quantum = torch.arange(1, 2 * order, 2, dtype=torch.float64, device=device)
    twice_harmonic = torch.diag((2 * quantum + 1).to(dtype))
    dimensionless_position_squared = odd_hermite_position_squared_matrix(
        order, 1.0, dtype=dtype, device=device
    )
    return (twice_harmonic - dimensionless_position_squared) / length_scale**2
