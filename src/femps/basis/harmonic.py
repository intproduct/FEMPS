"""Harmonic-oscillator functional basis used by Hong et al. (2022).

The matrix convention is ``operator[row=s_prime, column=s]``.  Therefore

    d/dx |s> = sqrt(s/2) |s-1> - sqrt((s+1)/2) |s+1>.

The finite basis is a Galerkin truncation. Canonical commutators and identities
that involve the omitted top state must only be asserted in the interior.
"""

from __future__ import annotations

import torch


def _validate_order(order: int) -> None:
    if order < 1:
        raise ValueError(f"basis order must be positive, got {order}")


def position_matrix(
    order: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the truncated position matrix ``<s'|x|s>``."""
    _validate_order(order)
    x = torch.zeros((order, order), dtype=dtype, device=device)
    if order == 1:
        return x
    s = torch.arange(1, order, dtype=torch.float64, device=device)
    values = torch.sqrt(s / 2).to(dtype=dtype)
    idx = torch.arange(order - 1, device=device)
    x[idx, idx + 1] = values
    x[idx + 1, idx] = values
    return x


def derivative_matrix(
    order: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the truncated derivative matrix ``<s'|d/dx|s>``."""
    _validate_order(order)
    derivative = torch.zeros((order, order), dtype=dtype, device=device)
    if order == 1:
        return derivative
    s = torch.arange(1, order, dtype=torch.float64, device=device)
    values = torch.sqrt(s / 2).to(dtype=dtype)
    idx = torch.arange(order - 1, device=device)
    derivative[idx, idx + 1] = values
    derivative[idx + 1, idx] = -values
    return derivative


def position_squared_matrix(
    order: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the Galerkin projection ``<s'|x^2|s>``.

    This is not ``position_matrix(order) ** 2``: the latter drops the virtual
    excursion through the first omitted basis state at the top boundary.
    """

    _validate_order(order)
    result = torch.zeros((order, order), dtype=dtype, device=device)
    number = torch.arange(order, dtype=torch.float64, device=device)
    result.diagonal().copy_((number + 0.5).to(dtype))
    if order > 2:
        lower = torch.arange(order - 2, dtype=torch.float64, device=device)
        values = 0.5 * torch.sqrt((lower + 1.0) * (lower + 2.0))
        index = torch.arange(order - 2, device=device)
        result[index, index + 2] = values.to(dtype)
        result[index + 2, index] = values.to(dtype)
    return result


def harmonic_hamiltonian(
    order: int,
    *,
    omega: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Projected ``-1/2 d^2/dx^2 + omega^2 x^2/2`` matrix.

    We project the infinite-basis operator term by term. The kinetic operator
    is constructed from ladder operators before truncation, avoiding the false
    top-state identity produced by squaring a finite derivative matrix.
    """
    _validate_order(order)
    x_squared = position_squared_matrix(order, dtype=dtype, device=device)
    number = torch.arange(order, dtype=torch.float64, device=device).to(dtype)
    # Construct H_0 exactly on the truncated eigenbasis, then add the projected
    # frequency shift. This avoids squaring a truncated derivative operator.
    h_unit = torch.diag(number + 0.5)
    return h_unit + 0.5 * (omega * omega - 1.0) * x_squared
