import math

import numpy as np
import pytest
import torch

from femps.basis.multiscale_odd_hermite import (
    multiscale_odd_hermite_basis_values,
    multiscale_odd_hermite_characteristic_matrices,
    multiscale_odd_hermite_condition_number,
    multiscale_odd_hermite_derivative_matrix,
    multiscale_odd_hermite_negative_second_derivative_matrix,
    multiscale_odd_hermite_position_matrix,
    multiscale_odd_hermite_position_squared_matrix,
)


def _quadrature(
    order: int,
    scale: float,
    ratio: float,
    count: int = 700,
):
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(count)
    maximum_mode = (order + 1) // 2 - 1
    maximum_scale = scale * math.sqrt(ratio)
    cutoff = maximum_scale * (math.sqrt(4 * (maximum_mode + 1) + 2) + 10)
    nodes = torch.from_numpy((raw_nodes + 1) * cutoff / 2)
    weights = torch.from_numpy(raw_weights * cutoff / 2)
    basis = multiscale_odd_hermite_basis_values(
        order, nodes, scale, ratio
    )
    return nodes, weights, basis


def test_multiscale_basis_is_orthonormal_and_collision_dirichlet() -> None:
    order = 6
    scale = 0.8
    ratio = 2.0
    nodes, weights, basis = _quadrature(order, scale, ratio)
    overlap = torch.einsum("xm,x,xn->mn", basis, weights, basis)
    torch.testing.assert_close(
        overlap, torch.eye(order, dtype=torch.float64), atol=2e-11, rtol=2e-11
    )
    boundary = multiscale_odd_hermite_basis_values(
        order, torch.zeros(1, dtype=torch.float64), scale, ratio
    )
    torch.testing.assert_close(boundary, torch.zeros_like(boundary), atol=0, rtol=0)
    assert multiscale_odd_hermite_condition_number(order, scale, ratio) < 1e8


def test_multiscale_analytic_local_operators_match_independent_quadrature() -> None:
    order = 5
    scale = 0.9
    ratio = 2.2
    nodes, weights, basis = _quadrature(order, scale, ratio)
    step = 2e-5
    plus = multiscale_odd_hermite_basis_values(
        order, nodes + step, scale, ratio
    )
    minus = multiscale_odd_hermite_basis_values(
        order, nodes - step, scale, ratio
    )
    derivatives = (plus - minus) / (2 * step)
    expected_derivative = torch.einsum(
        "xm,x,xn->mn", basis, weights, derivatives
    )
    expected_kinetic = torch.einsum(
        "xm,x,xn->mn", derivatives, weights, derivatives
    )
    expected_position = torch.einsum(
        "xm,x,x,xn->mn", basis, weights, nodes, basis
    )
    expected_squared = torch.einsum(
        "xm,x,x,xn->mn", basis, weights, nodes.square(), basis
    )
    torch.testing.assert_close(
        multiscale_odd_hermite_derivative_matrix(order, scale, ratio),
        expected_derivative,
        atol=2e-8,
        rtol=2e-8,
    )
    torch.testing.assert_close(
        multiscale_odd_hermite_negative_second_derivative_matrix(
            order, scale, ratio
        ),
        expected_kinetic,
        atol=4e-8,
        rtol=4e-8,
    )
    torch.testing.assert_close(
        multiscale_odd_hermite_position_matrix(order, scale, ratio),
        expected_position,
        atol=2e-11,
        rtol=2e-11,
    )
    torch.testing.assert_close(
        multiscale_odd_hermite_position_squared_matrix(order, scale, ratio),
        expected_squared,
        atol=2e-11,
        rtol=2e-11,
    )


def test_multiscale_characteristic_matrices_match_independent_quadrature() -> None:
    order = 5
    scale = 0.8
    ratio = 2.0
    frequencies = torch.tensor([0.0, 0.7, 2.1, 5.0], dtype=torch.float64)
    cosine, sine = multiscale_odd_hermite_characteristic_matrices(
        order,
        frequencies,
        scale,
        ratio,
        quadrature_order=180,
    )
    nodes, weights, basis = _quadrature(order, scale, ratio)
    phases = frequencies[:, None] * nodes[None]
    expected_cosine = torch.einsum(
        "xm,x,kx,xn->kmn", basis, weights, torch.cos(phases), basis
    )
    expected_sine = torch.einsum(
        "xm,x,kx,xn->kmn", basis, weights, torch.sin(phases), basis
    )
    torch.testing.assert_close(cosine, expected_cosine, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(sine, expected_sine, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(
        cosine[0], torch.eye(order, dtype=torch.float64), atol=2e-11, rtol=2e-11
    )


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_multiscale_basis_rejects_invalid_scale_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="scale_ratio"):
        multiscale_odd_hermite_condition_number(4, 0.8, ratio)
