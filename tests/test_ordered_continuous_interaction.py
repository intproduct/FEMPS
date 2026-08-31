import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.integration
pytest.importorskip("latticetn")

from femps.basis.dirichlet_sine import dirichlet_sine_basis_values
from femps.baselines.ordered_continuous_interaction import (
    ordered_continuous_soft_coulomb_pair_mpo,
    soft_coulomb_chebyshev_sampled_error,
)


def _quadrature(length: float, count: int = 120):
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(count)
    nodes = torch.from_numpy((raw_nodes + 1) * length / 2)
    weights = torch.from_numpy(raw_weights * length / 2)
    return nodes, weights


def _one_distance_truth(order: int, length: float, softening: float) -> torch.Tensor:
    nodes, weights = _quadrature(length)
    basis = dirichlet_sine_basis_values(order, nodes, length)
    potential = 1 / torch.sqrt(nodes**2 + softening**2)
    return torch.einsum("xm,x,x,xn->mn", basis, weights, potential, basis)


def _two_distance_truth(order: int, length: float, softening: float) -> torch.Tensor:
    nodes, weights = _quadrature(length, 90)
    basis = dirichlet_sine_basis_values(order, nodes, length)
    density = torch.einsum("xm,xn->xmn", basis, basis)
    potential = 1 / torch.sqrt(
        (nodes[:, None] + nodes[None, :]) ** 2 + softening**2
    )
    tensor = torch.einsum(
        "xmn,yab,xy,x,y->manb", density, density, potential, weights, weights
    )
    return tensor.reshape(order**2, order**2)


def test_chebyshev_scalar_error_decreases_independently() -> None:
    errors = [
        soft_coulomb_chebyshev_sampled_error(12.0, degree)
        for degree in (8, 16, 24)
    ]
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 2e-5


def test_one_distance_pair_mpo_matches_direct_projected_quadrature() -> None:
    order = 5
    length = 8.0
    expected = _one_distance_truth(order, length, 1.0)
    errors = []
    for degree in (8, 16):
        mpo = ordered_continuous_soft_coulomb_pair_mpo(
            2, order, length, 0, 1, degree, quadrature_order=120
        )
        observed = mpo.to_dense().reshape(order, order, order, order)[
            0, :, 0, :
        ]
        errors.append(float(torch.linalg.matrix_norm(observed - expected)))
    assert errors[1] < errors[0]
    assert errors[-1] < 2e-4


def test_two_distance_interval_mpo_matches_direct_multidimensional_quadrature() -> None:
    order = 3
    length = 6.0
    expected_relative = _two_distance_truth(order, length, 1.0)
    errors = []
    for degree in (8, 16):
        mpo = ordered_continuous_soft_coulomb_pair_mpo(
            3, order, length, 0, 2, degree, quadrature_order=120
        )
        expected = torch.kron(torch.eye(order), expected_relative)
        observed = mpo.to_dense()
        torch.testing.assert_close(observed.mT, observed, atol=3e-13, rtol=3e-13)
        errors.append(float(torch.linalg.matrix_norm(observed - expected)))
        assert max(max(tensor.shape[:2]) for tensor in mpo.tensors) <= degree + 1
    assert errors[1] < errors[0]
    assert errors[-1] < 2e-3
    assert math.isfinite(sum(errors))
