"""Orthonormalized two-scale odd-Hermite functions on the positive half-line."""

from __future__ import annotations

import math

import numpy as np
import torch

from femps.basis.odd_hermite import odd_hermite_basis_values


def _validate(order: int, length_scale: float, scale_ratio: float) -> None:
    if order < 1:
        raise ValueError("basis order must be positive")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")
    if scale_ratio <= 1:
        raise ValueError("scale_ratio must exceed one")


def _primitive_specifications(
    order: int,
    length_scale: float,
    scale_ratio: float,
) -> tuple[tuple[int, float], ...]:
    """Return odd-Hermite mode/scale pairs with a fixed geometric mean."""

    _validate(order, length_scale, scale_ratio)
    if order == 1:
        return ((0, length_scale),)
    short_count = (order + 1) // 2
    long_count = order - short_count
    root_ratio = math.sqrt(scale_ratio)
    short_scale = length_scale / root_ratio
    long_scale = length_scale * root_ratio
    return tuple(
        [(mode, short_scale) for mode in range(short_count)]
        + [(mode, long_scale) for mode in range(long_count)]
    )


def _primitive_polynomial(mode: int, scale: float) -> np.ndarray:
    """Return P where phi(r)=P(r)*exp(-r^2/(2*scale^2))."""

    quantum_number = 2 * mode + 1
    hermite_series = np.zeros(quantum_number + 1, dtype=np.float64)
    hermite_series[quantum_number] = 1
    coefficients = np.polynomial.hermite.herm2poly(hermite_series)
    normalization = math.sqrt(2 / scale) / (
        math.pi**0.25
        * math.sqrt(2**quantum_number * math.factorial(quantum_number))
    )
    powers = np.arange(coefficients.size, dtype=np.float64)
    return normalization * coefficients / scale**powers


def _derivative_polynomial(polynomial: np.ndarray, scale: float) -> np.ndarray:
    derivative = np.zeros(polynomial.size + 1, dtype=np.float64)
    if polynomial.size > 1:
        derivative[: polynomial.size - 1] = (
            np.arange(1, polynomial.size) * polynomial[1:]
        )
    derivative[1:] -= polynomial / scale**2
    return derivative


def _gaussian_polynomial_integral(
    left: np.ndarray,
    right: np.ndarray,
    gaussian_rate: float,
    power: int = 0,
) -> float:
    coefficients = np.convolve(left, right)
    result = 0.0
    for degree, coefficient in enumerate(coefficients):
        total_degree = degree + power
        result += (
            0.5
            * coefficient
            * gaussian_rate ** (-(total_degree + 1) / 2)
            * math.gamma((total_degree + 1) / 2)
        )
    return result


def _primitive_matrix(
    specifications: tuple[tuple[int, float], ...],
    *,
    power: int = 0,
    left_derivative: bool = False,
    right_derivative: bool = False,
) -> np.ndarray:
    polynomials = [
        _primitive_polynomial(mode, scale)
        for mode, scale in specifications
    ]
    left_polynomials = (
        [
            _derivative_polynomial(polynomial, scale)
            for polynomial, (_, scale) in zip(
                polynomials, specifications, strict=True
            )
        ]
        if left_derivative
        else polynomials
    )
    right_polynomials = (
        [
            _derivative_polynomial(polynomial, scale)
            for polynomial, (_, scale) in zip(
                polynomials, specifications, strict=True
            )
        ]
        if right_derivative
        else polynomials
    )
    matrix = np.empty(
        (len(specifications), len(specifications)), dtype=np.float64
    )
    for left_index, (_, left_scale) in enumerate(specifications):
        for right_index, (_, right_scale) in enumerate(specifications):
            gaussian_rate = 0.5 * (
                1 / left_scale**2 + 1 / right_scale**2
            )
            matrix[left_index, right_index] = _gaussian_polynomial_integral(
                left_polynomials[left_index],
                right_polynomials[right_index],
                gaussian_rate,
                power,
            )
    return matrix


def _orthogonalizer(
    order: int,
    length_scale: float,
    scale_ratio: float,
) -> tuple[tuple[tuple[int, float], ...], np.ndarray, np.ndarray]:
    specifications = _primitive_specifications(
        order, length_scale, scale_ratio
    )
    overlap = _primitive_matrix(specifications)
    overlap = 0.5 * (overlap + overlap.T)
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    if eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
        raise ValueError(
            "multiscale primitive overlap is numerically rank deficient; "
            "increase scale_ratio or reduce basis order"
        )
    inverse_square_root = (
        eigenvectors * eigenvalues[None, :] ** -0.5
    ) @ eigenvectors.T
    return specifications, inverse_square_root, eigenvalues


def multiscale_odd_hermite_condition_number(
    order: int,
    length_scale: float = 1.0,
    scale_ratio: float = 2.0,
) -> float:
    """Return the analytic primitive overlap condition number."""

    _, _, eigenvalues = _orthogonalizer(order, length_scale, scale_ratio)
    return float(eigenvalues[-1] / eigenvalues[0])


def multiscale_odd_hermite_basis_values(
    order: int,
    points: torch.Tensor,
    length_scale: float = 1.0,
    scale_ratio: float = 2.0,
) -> torch.Tensor:
    """Evaluate the symmetric-orthonormalized two-scale basis."""

    specifications, transform, _ = _orthogonalizer(
        order, length_scale, scale_ratio
    )
    columns = [
        odd_hermite_basis_values(mode + 1, points, scale)[:, mode]
        for mode, scale in specifications
    ]
    primitives = torch.stack(columns, dim=1)
    transform_tensor = torch.as_tensor(
        transform, dtype=points.dtype, device=points.device
    )
    return primitives @ transform_tensor


def _orthonormal_operator_matrix(
    order: int,
    length_scale: float,
    scale_ratio: float,
    *,
    power: int = 0,
    left_derivative: bool = False,
    right_derivative: bool = False,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    specifications, transform, _ = _orthogonalizer(
        order, length_scale, scale_ratio
    )
    primitive = _primitive_matrix(
        specifications,
        power=power,
        left_derivative=left_derivative,
        right_derivative=right_derivative,
    )
    result = transform.T @ primitive @ transform
    return torch.as_tensor(result, dtype=dtype, device=device)


def multiscale_odd_hermite_derivative_matrix(
    order: int,
    length_scale: float = 1.0,
    scale_ratio: float = 2.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the analytic projection ``<m|d/dr|n>``."""

    return _orthonormal_operator_matrix(
        order,
        length_scale,
        scale_ratio,
        right_derivative=True,
        dtype=dtype,
        device=device,
    )


def multiscale_odd_hermite_negative_second_derivative_matrix(
    order: int,
    length_scale: float = 1.0,
    scale_ratio: float = 2.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``<d phi_m/dr|d phi_n/dr>`` by analytic Gaussian moments."""

    return _orthonormal_operator_matrix(
        order,
        length_scale,
        scale_ratio,
        left_derivative=True,
        right_derivative=True,
        dtype=dtype,
        device=device,
    )


def multiscale_odd_hermite_position_matrix(
    order: int,
    length_scale: float = 1.0,
    scale_ratio: float = 2.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the analytic half-line position projection."""

    return _orthonormal_operator_matrix(
        order,
        length_scale,
        scale_ratio,
        power=1,
        dtype=dtype,
        device=device,
    )


def multiscale_odd_hermite_position_squared_matrix(
    order: int,
    length_scale: float = 1.0,
    scale_ratio: float = 2.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the analytic half-line squared-position projection."""

    return _orthonormal_operator_matrix(
        order,
        length_scale,
        scale_ratio,
        power=2,
        dtype=dtype,
        device=device,
    )


def multiscale_odd_hermite_characteristic_matrices(
    order: int,
    frequencies: torch.Tensor,
    length_scale: float = 1.0,
    scale_ratio: float = 2.0,
    *,
    quadrature_order: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cosine/sine projections with independent finite quadrature."""

    _validate(order, length_scale, scale_ratio)
    if frequencies.ndim != 1:
        raise ValueError("frequencies must be a one-dimensional tensor")
    resolved_device = frequencies.device if device is None else torch.device(device)
    count = max(256, 4 * order + 64) if quadrature_order is None else quadrature_order
    if count < 1:
        raise ValueError("quadrature_order must be positive")
    specifications = _primitive_specifications(order, length_scale, scale_ratio)
    maximum_mode = max(mode for mode, _ in specifications)
    maximum_scale = max(scale for _, scale in specifications)
    cutoff = maximum_scale * (math.sqrt(4 * (maximum_mode + 1) + 2) + 8)
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(count)
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    nodes = torch.as_tensor(
        (raw_nodes + 1) * cutoff / 2,
        dtype=real_dtype,
        device=resolved_device,
    )
    weights = torch.as_tensor(
        raw_weights * cutoff / 2,
        dtype=dtype,
        device=resolved_device,
    )
    basis = multiscale_odd_hermite_basis_values(
        order, nodes, length_scale, scale_ratio
    ).to(dtype)
    phases = (
        frequencies.to(dtype=dtype, device=resolved_device)[:, None]
        * nodes.to(dtype)[None]
    )
    cosine = torch.einsum(
        "xm,x,kx,xn->kmn",
        basis.conj(),
        weights,
        torch.cos(phases),
        basis,
    )
    sine = torch.einsum(
        "xm,x,kx,xn->kmn",
        basis.conj(),
        weights,
        torch.sin(phases),
        basis,
    )
    return cosine, sine
