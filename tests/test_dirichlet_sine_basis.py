import math

import numpy as np
import pytest
import torch

from femps.basis.dirichlet_sine import (
    dirichlet_sine_affine_power_matrices,
    dirichlet_sine_basis_values,
    dirichlet_sine_derivative_matrix,
    dirichlet_sine_negative_second_derivative_matrix,
    dirichlet_sine_position_matrix,
    dirichlet_sine_position_squared_matrix,
)


def _gauss_legendre(length: float, count: int = 300):
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(count)
    nodes = torch.from_numpy((raw_nodes + 1) * length / 2)
    weights = torch.from_numpy(raw_weights * length / 2)
    return nodes, weights


def test_sine_basis_is_orthonormal_and_dirichlet() -> None:
    order = 7
    length = 5.3
    nodes, weights = _gauss_legendre(length)
    values = dirichlet_sine_basis_values(order, nodes, length)
    overlap = torch.einsum("xm,x,xn->mn", values, weights, values)
    torch.testing.assert_close(
        overlap, torch.eye(order, dtype=torch.float64), atol=4e-14, rtol=4e-14
    )
    boundaries = dirichlet_sine_basis_values(
        order, torch.tensor([0.0, length], dtype=torch.float64), length
    )
    torch.testing.assert_close(
        boundaries, torch.zeros_like(boundaries), atol=2e-15, rtol=0
    )


def test_derivative_matrix_matches_independent_quadrature_and_is_skew() -> None:
    order = 6
    length = 4.7
    nodes, weights = _gauss_legendre(length)
    values = dirichlet_sine_basis_values(order, nodes, length)
    modes = torch.arange(1, order + 1, dtype=torch.float64)
    derivatives = (
        math.sqrt(2 / length)
        * (math.pi * modes / length)
        * torch.cos(math.pi * nodes[:, None] * modes / length)
    )
    expected = torch.einsum("xm,x,xn->mn", values, weights, derivatives)
    observed = dirichlet_sine_derivative_matrix(order, length)
    torch.testing.assert_close(observed, expected, atol=8e-14, rtol=8e-14)
    torch.testing.assert_close(observed.mT, -observed, atol=0, rtol=0)


def test_negative_second_derivative_has_exact_box_spectrum() -> None:
    order = 8
    length = 3.2
    observed = dirichlet_sine_negative_second_derivative_matrix(order, length)
    expected = (
        math.pi * torch.arange(1, order + 1, dtype=torch.float64) / length
    ) ** 2
    torch.testing.assert_close(observed.diagonal(), expected)
    torch.testing.assert_close(
        observed - torch.diag(observed.diagonal()), torch.zeros_like(observed)
    )


def test_position_matrices_match_independent_quadrature() -> None:
    order = 7
    length = 6.1
    nodes, weights = _gauss_legendre(length)
    values = dirichlet_sine_basis_values(order, nodes, length)
    expected_position = torch.einsum(
        "xm,x,x,xn->mn", values, weights, nodes, values
    )
    expected_squared = torch.einsum(
        "xm,x,x,xn->mn", values, weights, nodes**2, values
    )
    torch.testing.assert_close(
        dirichlet_sine_position_matrix(order, length),
        expected_position,
        atol=2e-13,
        rtol=2e-13,
    )
    torch.testing.assert_close(
        dirichlet_sine_position_squared_matrix(order, length),
        expected_squared,
        atol=8e-13,
        rtol=8e-13,
    )


@pytest.mark.parametrize("order,length", [(0, 2.0), (3, 0.0), (3, -1.0)])
def test_invalid_sine_basis_parameters_are_rejected(order: int, length: float) -> None:
    with pytest.raises(ValueError):
        dirichlet_sine_position_matrix(order, length)


def test_affine_power_quadrature_reproduces_analytic_low_powers() -> None:
    order = 6
    length = 5.0
    powers = dirichlet_sine_affine_power_matrices(
        order, length, 2, quadrature_order=100
    )
    torch.testing.assert_close(
        powers[0], torch.eye(order, dtype=torch.float64), atol=4e-14, rtol=4e-14
    )
    torch.testing.assert_close(
        powers[1],
        dirichlet_sine_position_matrix(order, length),
        atol=2e-13,
        rtol=2e-13,
    )
    torch.testing.assert_close(
        powers[2],
        dirichlet_sine_position_squared_matrix(order, length),
        atol=8e-13,
        rtol=8e-13,
    )
