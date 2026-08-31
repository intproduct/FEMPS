"""Orthonormal Dirichlet sine basis on a finite positive interval."""

from __future__ import annotations

import math

import numpy as np
import torch


def _validate(order: int, length: float) -> None:
    if order < 1:
        raise ValueError("basis order must be positive")
    if length <= 0:
        raise ValueError("interval length must be positive")


def dirichlet_sine_basis_values(
    order: int,
    points: torch.Tensor,
    length: float,
) -> torch.Tensor:
    """Evaluate ``sqrt(2/R) sin((n+1) pi r/R)`` at tensor ``points``."""

    _validate(order, length)
    modes = torch.arange(
        1, order + 1, dtype=points.real.dtype, device=points.device
    )
    arguments = math.pi * points[..., None] * modes / length
    return math.sqrt(2 / length) * torch.sin(arguments).to(points.dtype)


def dirichlet_sine_derivative_matrix(
    order: int,
    length: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the Galerkin matrix ``<m|d/dr|n>``."""

    _validate(order, length)
    result = torch.zeros(order, order, dtype=dtype, device=device)
    for row in range(order):
        for column in range(order):
            if (row + column) % 2 == 0:
                continue
            left = row + 1
            right = column + 1
            result[row, column] = (
                4 * left * right / (length * (left * left - right * right))
            )
    return result


def dirichlet_sine_negative_second_derivative_matrix(
    order: int,
    length: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``<m|-d^2/dr^2|n>`` (diagonal in the sine basis)."""

    _validate(order, length)
    modes = torch.arange(1, order + 1, dtype=torch.float64, device=device)
    eigenvalues = (math.pi * modes / length) ** 2
    return torch.diag(eigenvalues.to(dtype))


def dirichlet_sine_position_matrix(
    order: int,
    length: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the exact Galerkin matrix ``<m|r|n>``."""

    _validate(order, length)
    result = torch.zeros(order, order, dtype=dtype, device=device)
    result.diagonal().fill_(length / 2)
    coefficient = 2 * length / math.pi**2
    for row in range(order):
        for column in range(row + 1, order):
            if (row + column) % 2 == 0:
                continue
            difference = row - column
            total = row + column + 2
            value = coefficient * (
                1 / total**2 - 1 / difference**2
            )
            result[row, column] = value
            result[column, row] = value
    return result


def dirichlet_sine_position_squared_matrix(
    order: int,
    length: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the exact Galerkin matrix ``<m|r^2|n>``."""

    _validate(order, length)
    result = torch.zeros(order, order, dtype=dtype, device=device)
    coefficient = 2 * length**2 / math.pi**2
    for row in range(order):
        mode = row + 1
        result[row, row] = length**2 * (
            1 / 3 - 1 / (2 * math.pi**2 * mode**2)
        )
        for column in range(row + 1, order):
            difference = row - column
            total = row + column + 2
            parity = 1 if (row + column) % 2 == 0 else -1
            value = coefficient * parity * (
                1 / difference**2 - 1 / total**2
            )
            result[row, column] = value
            result[column, row] = value
    return result


def dirichlet_sine_affine_power_matrices(
    order: int,
    length: float,
    maximum_power: int,
    *,
    scale: float = 1.0,
    shift: float = 0.0,
    quadrature_order: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return Galerkin matrices of ``(scale*r+shift)^p`` for ``0<=p<=P``.

    Gauss--Legendre quadrature is used deliberately here because later
    interval-potential polynomials require powers beyond ``r^2``.  Callers that
    use these matrices as numerical evidence must vary ``quadrature_order``.
    """

    _validate(order, length)
    if maximum_power < 0:
        raise ValueError("maximum_power must be nonnegative")
    count = (
        max(64, 2 * order + maximum_power + 8)
        if quadrature_order is None
        else quadrature_order
    )
    if count < 1:
        raise ValueError("quadrature_order must be positive")
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(count)
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    nodes = torch.as_tensor(
        (raw_nodes + 1) * length / 2,
        dtype=real_dtype,
        device=device,
    )
    weights = torch.as_tensor(
        raw_weights * length / 2,
        dtype=real_dtype,
        device=device,
    )
    basis = dirichlet_sine_basis_values(order, nodes, length).to(dtype)
    affine = (scale * nodes + shift).to(dtype)
    matrices = []
    power = torch.ones_like(affine)
    for _ in range(maximum_power + 1):
        matrices.append(
            torch.einsum("xm,x,x,xn->mn", basis.conj(), weights, power, basis)
        )
        power = power * affine
    return tuple(matrices)
