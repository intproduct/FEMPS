import torch

from femps.basis.harmonic import (
    derivative_matrix,
    harmonic_hamiltonian,
    negative_second_derivative_matrix,
    position_matrix,
    position_squared_matrix,
)


def test_position_and_derivative_adjoint_symmetries():
    x = position_matrix(8)
    derivative = derivative_matrix(8)
    torch.testing.assert_close(x.T, x)
    torch.testing.assert_close(derivative.T, -derivative)


def test_derivative_ladder_coefficients():
    derivative = derivative_matrix(5)
    expected = torch.tensor(0.5, dtype=torch.float64).sqrt()
    torch.testing.assert_close(derivative[0, 1], expected)
    torch.testing.assert_close(derivative[1, 0], -expected)
    torch.testing.assert_close(
        derivative[1, 2], torch.tensor(1.0, dtype=torch.float64)
    )


def test_unit_harmonic_hamiltonian_has_exact_low_spectrum():
    hamiltonian = harmonic_hamiltonian(8)
    expected = torch.arange(8, dtype=torch.float64) + 0.5
    torch.testing.assert_close(torch.diagonal(hamiltonian), expected)
    torch.testing.assert_close(
        hamiltonian - torch.diag(torch.diagonal(hamiltonian)),
        torch.zeros_like(hamiltonian),
    )


def test_position_squared_uses_infinite_basis_projection_at_top_boundary():
    order = 6
    squared = position_squared_matrix(order)
    expected_diagonal = torch.arange(order, dtype=torch.float64) + 0.5
    torch.testing.assert_close(squared.diagonal(), expected_diagonal)
    assert squared[-1, -1] != (position_matrix(order) @ position_matrix(order))[-1, -1]


def test_negative_second_derivative_is_projected_before_truncation():
    order = 7
    kinetic = negative_second_derivative_matrix(order)
    squared = position_squared_matrix(order)
    expected = torch.diag(2 * torch.arange(order, dtype=torch.float64) + 1)
    torch.testing.assert_close(kinetic + squared, expected)
    torch.testing.assert_close(kinetic.mT, kinetic)
    assert kinetic[-1, -1] != -(derivative_matrix(order) @ derivative_matrix(order))[-1, -1]
