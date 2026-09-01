import torch

from femps.algorithms import (
    canonical_slater_orbitals,
    select_adaptive_diagonal_path_term,
    solve_generalized_hermitian,
)
from femps.exterior import (
    antisymmetry_residual,
    diagonal_path_exterior_coefficients,
    diagonal_path_hamiltonian_matrices,
    exterior_coefficients_to_tensor,
)
from femps.hamiltonians import (
    FactorizedTwoBodyOperator,
    antisymmetric_many_body_hamiltonian,
)


def _complex_random(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(shape, generator=generator, dtype=torch.float64),
        torch.randn(shape, generator=generator, dtype=torch.float64),
    )


def _growth_problem() -> tuple[
    torch.Tensor, torch.Tensor, FactorizedTwoBodyOperator
]:
    dimension, particles = 5, 3
    source = canonical_slater_orbitals(
        _complex_random((2, dimension, particles), 3401)
    )
    raw_one = _complex_random((dimension, dimension), 3402)
    one_body = 0.5 * (raw_one + raw_one.mH)
    raw_left = _complex_random((2, dimension, dimension), 3403)
    raw_right = _complex_random((2, dimension, dimension), 3404)
    interaction = FactorizedTwoBodyOperator(
        0.5 * (raw_left + raw_left.mH),
        0.5 * (raw_right + raw_right.mH),
        torch.tensor([0.11, -0.07], dtype=torch.complex128),
    )
    return source, one_body, interaction


def _explicit_matrices(
    orbitals: torch.Tensor,
    truth_hamiltonian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    columns = torch.stack(
        [
            diagonal_path_exterior_coefficients(
                orbitals[index : index + 1],
                torch.ones(1, dtype=orbitals.dtype, device=orbitals.device),
            )
            for index in range(orbitals.shape[0])
        ],
        dim=1,
    )
    return columns.mH @ columns, columns.mH @ truth_hamiltonian @ columns


def test_adaptive_growth_is_seeded_nested_and_truth_free() -> None:
    source, one_body, interaction = _growth_problem()
    growth = select_adaptive_diagonal_path_term(
        source,
        one_body,
        interaction,
        pool_size=8,
        seed=3410,
    )
    repeated = select_adaptive_diagonal_path_term(
        source,
        one_body,
        interaction,
        pool_size=8,
        seed=3410,
    )

    assert torch.equal(growth.orbitals, repeated.orbitals)
    assert growth.selected_candidate == repeated.selected_candidate
    assert growth.source_terms == 2
    assert growth.orbitals.shape == (3, 5, 3)
    torch.testing.assert_close(growth.orbitals[:2], source, atol=3e-14, rtol=3e-14)
    assert growth.predicted_energy <= growth.source_energy + 1e-10
    assert len(growth.candidates) == 8
    assert growth.candidates[growth.selected_candidate].admitted

    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        growth.orbitals,
        one_body,
        two_body_left=interaction.left,
        two_body_right=interaction.right,
        two_body_weights=interaction.weights,
    )
    polynomial = solve_generalized_hermitian(hamiltonian, overlap)
    truth_hamiltonian = antisymmetric_many_body_hamiltonian(
        one_body, 3, interaction
    )
    explicit_overlap, explicit_hamiltonian = _explicit_matrices(
        growth.orbitals, truth_hamiltonian
    )
    explicit = solve_generalized_hermitian(explicit_hamiltonian, explicit_overlap)
    torch.testing.assert_close(polynomial.energy, explicit.energy, atol=1e-10, rtol=1e-10)
    assert abs(growth.predicted_energy - float(explicit.energy)) < 1e-10

    coefficients = diagonal_path_exterior_coefficients(
        growth.orbitals, explicit.amplitudes
    )
    particle_state = exterior_coefficients_to_tensor(coefficients, 5, 3)
    assert float(antisymmetry_residual(particle_state)) < 1e-12


def test_adaptive_growth_selected_state_gradient_matches_exterior_truth() -> None:
    source, one_body, interaction = _growth_problem()
    selected = select_adaptive_diagonal_path_term(
        source,
        one_body,
        interaction,
        pool_size=6,
        seed=3411,
    ).orbitals
    truth_hamiltonian = antisymmetric_many_body_hamiltonian(
        one_body, 3, interaction
    )

    polynomial_orbitals = selected.clone().requires_grad_(True)
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        polynomial_orbitals,
        one_body,
        two_body_left=interaction.left,
        two_body_right=interaction.right,
        two_body_weights=interaction.weights,
    )
    polynomial_energy = solve_generalized_hermitian(hamiltonian, overlap).energy
    polynomial_gradient = torch.autograd.grad(
        polynomial_energy, polynomial_orbitals
    )[0]

    explicit_orbitals = selected.clone().requires_grad_(True)
    explicit_overlap, explicit_hamiltonian = _explicit_matrices(
        explicit_orbitals, truth_hamiltonian
    )
    explicit_energy = solve_generalized_hermitian(
        explicit_hamiltonian, explicit_overlap
    ).energy
    explicit_gradient = torch.autograd.grad(explicit_energy, explicit_orbitals)[0]

    torch.testing.assert_close(
        polynomial_energy, explicit_energy, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        polynomial_gradient, explicit_gradient, atol=1e-8, rtol=1e-8
    )
