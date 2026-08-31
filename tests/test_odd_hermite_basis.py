import numpy as np
import pytest
import torch

from femps.basis.odd_hermite import (
    odd_hermite_basis_values,
    odd_hermite_derivative_matrix,
    odd_hermite_negative_second_derivative_matrix,
    odd_hermite_position_matrix,
    odd_hermite_position_squared_matrix,
    odd_hermite_power_matrices,
)


def _finite_quadrature(length_scale: float, count: int = 500):
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(count)
    cutoff = 12 * length_scale
    nodes = torch.from_numpy((raw_nodes + 1) * cutoff / 2)
    weights = torch.from_numpy(raw_weights * cutoff / 2)
    return nodes, weights


def test_odd_hermite_is_orthonormal_dirichlet_and_unbounded() -> None:
    order = 6
    scale = 1.3
    nodes, weights = _finite_quadrature(scale)
    basis = odd_hermite_basis_values(order, nodes, scale)
    overlap = torch.einsum("xm,x,xn->mn", basis, weights, basis)
    torch.testing.assert_close(
        overlap, torch.eye(order, dtype=torch.float64), atol=2e-13, rtol=2e-13
    )
    boundary = odd_hermite_basis_values(
        order, torch.zeros(1, dtype=torch.float64), scale
    )
    torch.testing.assert_close(boundary, torch.zeros_like(boundary), atol=0, rtol=0)


def test_odd_hermite_power_quadrature_matches_independent_finite_integral() -> None:
    order = 5
    scale = 0.9
    nodes, weights = _finite_quadrature(scale)
    basis = odd_hermite_basis_values(order, nodes, scale)
    powers = odd_hermite_power_matrices(order, 2, scale, quadrature_order=128)
    for power in range(3):
        expected = torch.einsum(
            "xm,x,x,xn->mn", basis, weights, nodes**power, basis
        )
        torch.testing.assert_close(
            powers[power], expected, atol=3e-11, rtol=3e-11
        )
    torch.testing.assert_close(
        powers[2],
        odd_hermite_position_squared_matrix(order, scale),
        atol=3e-11,
        rtol=3e-11,
    )


def test_odd_hermite_derivative_is_nonzero_skew_and_matches_finite_difference() -> None:
    order = 5
    scale = 1.1
    nodes, weights = _finite_quadrature(scale)
    step = 2e-5
    plus = odd_hermite_basis_values(order, nodes + step, scale)
    minus = odd_hermite_basis_values(order, nodes - step, scale)
    derivative_values = (plus - minus) / (2 * step)
    basis = odd_hermite_basis_values(order, nodes, scale)
    expected = torch.einsum("xm,x,xn->mn", basis, weights, derivative_values)
    observed = odd_hermite_derivative_matrix(
        order, scale, quadrature_order=128
    )
    assert float(torch.linalg.matrix_norm(observed)) > 0
    torch.testing.assert_close(observed.mT, -observed, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(observed, expected, atol=3e-9, rtol=3e-9)


def test_odd_hermite_kinetic_and_position_squared_rebuild_oscillator() -> None:
    order = 7
    scale = 1.4
    kinetic = odd_hermite_negative_second_derivative_matrix(order, scale)
    squared = odd_hermite_position_squared_matrix(order, scale)
    quantum = torch.arange(1, 2 * order, 2, dtype=torch.float64)
    oscillator = 0.5 * scale**2 * kinetic + 0.5 / scale**2 * squared
    torch.testing.assert_close(oscillator, torch.diag(quantum + 0.5))
    torch.testing.assert_close(kinetic.mT, kinetic)


@pytest.mark.parametrize("order,scale", [(0, 1.0), (3, 0.0), (3, -1.0)])
def test_invalid_odd_hermite_parameters_are_rejected(order: int, scale: float) -> None:
    with pytest.raises(ValueError):
        odd_hermite_position_matrix(order, scale)
