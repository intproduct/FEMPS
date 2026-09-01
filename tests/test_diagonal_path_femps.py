import math

import torch

from femps.basis import harmonic_hamiltonian
from femps.exterior import (
    antisymmetry_residual,
    diagonal_path_energy,
    diagonal_path_exterior_coefficients,
    diagonal_path_hamiltonian_matrices,
    diagonal_path_norm,
    diagonal_path_structural_counts,
    diagonal_path_transition_diagnostics,
    exterior_coefficients_to_tensor,
    materialize_femps_matrix,
    slater_sum_cores,
)
from femps.hamiltonians import (
    FactorizedTwoBodyOperator,
    antisymmetric_many_body_hamiltonian,
)


def _random_complex(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imag = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imag)


def _problem() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    FactorizedTwoBodyOperator,
]:
    terms, dimension, particles = 2, 5, 3
    orbitals = _random_complex((terms, dimension, particles), 2801)
    amplitudes = _random_complex((terms,), 2802)
    raw_one = _random_complex((dimension, dimension), 2803)
    one_body = 0.5 * (raw_one + raw_one.mH)
    raw_left = _random_complex((2, dimension, dimension), 2804)
    raw_right = _random_complex((2, dimension, dimension), 2805)
    left = 0.5 * (raw_left + raw_left.mH)
    right = 0.5 * (raw_right + raw_right.mH)
    weights = torch.tensor([0.17, -0.09], dtype=torch.complex128)
    return orbitals, amplitudes, one_body, FactorizedTwoBodyOperator(
        left, right, weights
    )


def test_diagonal_path_embedding_and_coefficients_agree() -> None:
    orbitals, amplitudes, _, _ = _problem()
    coefficients = diagonal_path_exterior_coefficients(orbitals, amplitudes)
    by_coefficients = exterior_coefficients_to_tensor(
        coefficients, orbitals.shape[1], orbitals.shape[2]
    )
    by_femps = materialize_femps_matrix(slater_sum_cores(orbitals, amplitudes))
    torch.testing.assert_close(by_coefficients, by_femps, atol=3e-12, rtol=3e-12)
    assert antisymmetry_residual(by_femps).item() < 2e-14


def test_diagonal_path_transitions_match_full_exterior_hamiltonian() -> None:
    orbitals, amplitudes, one_body, two_body = _problem()
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        orbitals,
        one_body,
        two_body_left=two_body.left,
        two_body_right=two_body.right,
        two_body_weights=two_body.weights,
    )
    coefficient_columns = torch.stack(
        [
            diagonal_path_exterior_coefficients(
                orbitals[index : index + 1],
                torch.ones(1, dtype=orbitals.dtype),
            )
            for index in range(orbitals.shape[0])
        ],
        dim=1,
    )
    truth_hamiltonian = antisymmetric_many_body_hamiltonian(
        one_body, orbitals.shape[2], two_body
    )
    expected_overlap = coefficient_columns.mH @ coefficient_columns
    expected_hamiltonian = coefficient_columns.mH @ truth_hamiltonian @ coefficient_columns
    torch.testing.assert_close(overlap, expected_overlap, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(
        hamiltonian, expected_hamiltonian, atol=3e-10, rtol=3e-10
    )
    expected_energy = (
        torch.vdot(amplitudes, expected_hamiltonian @ amplitudes)
        / torch.vdot(amplitudes, expected_overlap @ amplitudes)
    ).real
    observed_energy = diagonal_path_energy(
        orbitals,
        amplitudes,
        one_body,
        two_body_left=two_body.left,
        two_body_right=two_body.right,
        two_body_weights=two_body.weights,
    )
    torch.testing.assert_close(observed_energy, expected_energy, atol=3e-10, rtol=3e-10)
    minor_energy = diagonal_path_energy(
        orbitals,
        amplitudes,
        one_body,
        two_body_left=two_body.left,
        two_body_right=two_body.right,
        two_body_weights=two_body.weights,
        transition_algorithm="minor",
    )
    torch.testing.assert_close(observed_energy, minor_energy, atol=3e-10, rtol=3e-10)


def test_diagonal_path_gradients_match_exterior_truth() -> None:
    orbitals, amplitudes, one_body, two_body = _problem()
    orbitals = orbitals.detach().requires_grad_(True)
    amplitudes = amplitudes.detach().requires_grad_(True)
    observed = diagonal_path_energy(
        orbitals,
        amplitudes,
        one_body,
        two_body_left=two_body.left,
        two_body_right=two_body.right,
        two_body_weights=two_body.weights,
    )
    observed_gradients = torch.autograd.grad(observed, (orbitals, amplitudes))

    truth_orbitals = orbitals.detach().clone().requires_grad_(True)
    truth_amplitudes = amplitudes.detach().clone().requires_grad_(True)
    coefficients = diagonal_path_exterior_coefficients(
        truth_orbitals, truth_amplitudes
    )
    truth_hamiltonian = antisymmetric_many_body_hamiltonian(
        one_body, orbitals.shape[2], two_body
    )
    expected = (
        torch.vdot(coefficients, truth_hamiltonian @ coefficients)
        / torch.vdot(coefficients, coefficients)
    ).real
    expected_gradients = torch.autograd.grad(
        expected, (truth_orbitals, truth_amplitudes)
    )
    for observed_gradient, expected_gradient in zip(
        observed_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            observed_gradient, expected_gradient, atol=2e-9, rtol=2e-9
        )


def test_diagonal_path_orbital_gradient_matches_central_difference() -> None:
    orbitals, amplitudes, one_body, two_body = _problem()
    orbitals = orbitals.detach().requires_grad_(True)
    energy = diagonal_path_energy(
        orbitals,
        amplitudes,
        one_body,
        two_body_left=two_body.left,
        two_body_right=two_body.right,
        two_body_weights=two_body.weights,
    )
    gradient = torch.autograd.grad(energy, orbitals)[0]
    index = (1, 3, 2)
    epsilon = 1e-6
    plus = orbitals.detach().clone()
    minus = orbitals.detach().clone()
    plus[index] += epsilon
    minus[index] -= epsilon
    finite_difference = (
        diagonal_path_energy(
            plus,
            amplitudes,
            one_body,
            two_body_left=two_body.left,
            two_body_right=two_body.right,
            two_body_weights=two_body.weights,
        )
        - diagonal_path_energy(
            minus,
            amplitudes,
            one_body,
            two_body_left=two_body.left,
            two_body_right=two_body.right,
            two_body_weights=two_body.weights,
        )
    ) / (2 * epsilon)
    torch.testing.assert_close(
        gradient[index].real, finite_difference, atol=2e-7, rtol=2e-6
    )


def test_singular_cross_overlap_remains_exact() -> None:
    orbitals = torch.zeros((2, 4, 2), dtype=torch.float64)
    orbitals[0, 0, 0] = orbitals[0, 1, 1] = 1.0
    orbitals[1, 0, 0] = orbitals[1, 2, 1] = 1.0
    amplitudes = torch.tensor([0.4, -0.7], dtype=torch.float64)
    one_body = torch.diag(torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float64))
    observed = diagonal_path_energy(orbitals, amplitudes, one_body)
    coefficients = diagonal_path_exterior_coefficients(orbitals, amplitudes)
    truth = antisymmetric_many_body_hamiltonian(one_body, 2)
    expected = (
        torch.vdot(coefficients, truth @ coefficients)
        / torch.vdot(coefficients, coefficients)
    ).real
    torch.testing.assert_close(observed, expected, atol=1e-13, rtol=1e-13)
    diagnostics = diagonal_path_transition_diagnostics(orbitals)
    assert diagnostics["well_conditioned_inverse_pairs"] == 2
    assert diagnostics["singular_safe_minor_pairs"] == 2


def test_well_conditioned_transition_pairs_use_inverse_path() -> None:
    orbitals, _, _, _ = _problem()
    diagnostics = diagonal_path_transition_diagnostics(orbitals)
    assert diagnostics["well_conditioned_inverse_pairs"] == 4
    assert diagnostics["singular_safe_minor_pairs"] == 0


def test_noninteracting_harmonic_slater_has_k_one_and_exact_energy() -> None:
    particles, dimension = 4, 7
    orbitals = torch.eye(dimension, dtype=torch.float64)[:, :particles].unsqueeze(0)
    amplitudes = torch.ones(1, dtype=torch.float64)
    one_body = harmonic_hamiltonian(dimension, dtype=torch.float64)
    energy = diagonal_path_energy(orbitals, amplitudes, one_body)
    torch.testing.assert_close(
        energy, torch.tensor(0.5 * particles**2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        diagonal_path_norm(orbitals, amplitudes),
        torch.tensor(1.0, dtype=torch.float64),
    )
    counts = diagonal_path_structural_counts(particles, dimension, 1)
    assert counts["transition_pairs"] == 1
    assert counts["enumerated_virtual_paths"] == 0
    assert counts["materialized_particle_coefficients"] == 0
    assert math.comb(dimension, particles) > counts["transition_pairs"]
