import pytest
import torch

from femps.algorithms import solve_generalized_hermitian
from femps.exterior import agp_exterior_coefficients
from femps.hamiltonians import (
    agp_energy,
    agp_hamiltonian_transition_matrices,
    agp_sum_energy,
    antisymmetric_many_body_hamiltonian,
    harmonic_pair_hamiltonian,
)


def _pair_matrices(terms: int = 3, dimension: int = 6) -> torch.Tensor:
    generator = torch.Generator().manual_seed(601)
    real = torch.randn(
        terms, dimension, dimension, generator=generator, dtype=torch.float64
    )
    imaginary = torch.randn(
        terms, dimension, dimension, generator=generator, dtype=torch.float64
    )
    raw = torch.complex(real, imaginary)
    return (raw - raw.transpose(1, 2)) / dimension**0.5


def test_agp_transition_matrices_match_exterior_truth() -> None:
    pair_matrices = _pair_matrices()
    one_body, interaction = harmonic_pair_hamiltonian(6, kappa=0.2)
    overlap, hamiltonian = agp_hamiltonian_transition_matrices(
        pair_matrices, 2, one_body, interaction
    )
    coefficients = torch.stack(
        [agp_exterior_coefficients(matrix, 2) for matrix in pair_matrices]
    )
    exterior_hamiltonian = antisymmetric_many_body_hamiltonian(
        one_body, 4, interaction
    )
    expected_overlap = coefficients.conj() @ coefficients.transpose(0, 1)
    expected_hamiltonian = coefficients.conj() @ (
        exterior_hamiltonian @ coefficients.transpose(0, 1)
    )
    torch.testing.assert_close(overlap, expected_overlap, atol=4e-10, rtol=4e-11)
    torch.testing.assert_close(
        hamiltonian, expected_hamiltonian, atol=2e-8, rtol=2e-10
    )
    torch.testing.assert_close(
        overlap, overlap.conj().transpose(0, 1), atol=4e-10, rtol=4e-11
    )
    torch.testing.assert_close(
        hamiltonian,
        hamiltonian.conj().transpose(0, 1),
        atol=2e-8,
        rtol=2e-10,
    )


def test_conditioned_amplitude_solve_matches_agp_sum_rayleigh_quotient() -> None:
    pair_matrices = _pair_matrices()
    one_body, interaction = harmonic_pair_hamiltonian(6, kappa=0.2)
    overlap, hamiltonian = agp_hamiltonian_transition_matrices(
        pair_matrices, 2, one_body, interaction
    )
    result = solve_generalized_hermitian(hamiltonian, overlap)
    rayleigh = agp_sum_energy(
        pair_matrices, result.amplitudes, 2, one_body, interaction
    )
    torch.testing.assert_close(result.energy, rayleigh, atol=3e-11, rtol=3e-11)
    assert result.retained_rank == 3
    assert result.discarded_rank == 0
    assert result.retained_condition_number >= 1
    assert result.residual_norm < 2e-8


def test_conditioned_solve_discards_duplicate_agp_direction() -> None:
    pair = _pair_matrices(terms=1)[0]
    pair_matrices = torch.stack((pair, pair))
    one_body, interaction = harmonic_pair_hamiltonian(6, kappa=0.2)
    overlap, hamiltonian = agp_hamiltonian_transition_matrices(
        pair_matrices, 2, one_body, interaction
    )
    result = solve_generalized_hermitian(hamiltonian, overlap)
    expected = agp_energy(pair, 2, one_body, interaction)
    torch.testing.assert_close(result.energy, expected, atol=3e-11, rtol=3e-11)
    assert result.retained_rank == 1
    assert result.discarded_rank == 1


def test_conditioned_solve_rejects_invalid_threshold() -> None:
    identity = torch.eye(2, dtype=torch.complex128)
    with pytest.raises(ValueError, match="nonnegative"):
        solve_generalized_hermitian(identity, identity, relative_threshold=-1)
