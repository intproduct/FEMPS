import pytest
import torch

from femps.algorithms import (
    assess_term_pruning,
    contribution_gram_spectrum,
    leave_one_out_energies,
    overlap_whitening,
    solve_generalized_hermitian,
)
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


def test_overlap_whitening_preserves_explicit_state() -> None:
    pair_matrices = _pair_matrices()
    coefficients = torch.stack(
        [agp_exterior_coefficients(matrix, 2) for matrix in pair_matrices]
    )
    overlap = coefficients.conj() @ coefficients.transpose(0, 1)
    whitening = overlap_whitening(overlap)
    torch.testing.assert_close(
        whitening.transformation.mH @ overlap @ whitening.transformation,
        torch.eye(3, dtype=overlap.dtype),
        atol=2e-12,
        rtol=2e-12,
    )
    orthonormal_states = whitening.transformation.transpose(0, 1) @ coefficients
    coordinates = torch.tensor([0.2 + 0.1j, -0.3j, 0.7], dtype=overlap.dtype)
    term_amplitudes = whitening.transformation @ coordinates
    torch.testing.assert_close(
        term_amplitudes @ coefficients,
        coordinates @ orthonormal_states,
        atol=2e-12,
        rtol=2e-12,
    )


def test_overlap_whitening_is_invariant_to_term_scale_and_phase_gauges() -> None:
    pair_matrices = _pair_matrices()
    coefficients = torch.stack(
        [agp_exterior_coefficients(matrix, 2) for matrix in pair_matrices]
    )
    overlap = coefficients.conj() @ coefficients.transpose(0, 1)
    gauges = torch.tensor([100.0j, -0.02, 1.3 - 0.4j], dtype=overlap.dtype)
    gauged_overlap = gauges.conj()[:, None] * overlap * gauges[None, :]
    original = overlap_whitening(overlap)
    gauged = overlap_whitening(gauged_overlap)
    torch.testing.assert_close(
        gauged.eigenvalues, original.eigenvalues, atol=2e-12, rtol=2e-12
    )
    assert gauged.retained_rank == original.retained_rank
    torch.testing.assert_close(
        gauged.transformation.mH @ gauged_overlap @ gauged.transformation,
        torch.eye(3, dtype=overlap.dtype),
        atol=2e-12,
        rtol=2e-12,
    )


def test_contribution_spectrum_is_scale_phase_and_permutation_invariant() -> None:
    pair_matrices = _pair_matrices()
    coefficients = torch.stack(
        [agp_exterior_coefficients(matrix, 2) for matrix in pair_matrices]
    )
    overlap = coefficients.conj() @ coefficients.transpose(0, 1)
    amplitudes = torch.tensor([0.7, -0.2j, 0.3 + 0.1j], dtype=overlap.dtype)
    expected = contribution_gram_spectrum(overlap, amplitudes)
    gauges = torch.tensor([2.0j, -0.5, 1.3 - 0.4j], dtype=overlap.dtype)
    gauged_overlap = gauges.conj()[:, None] * overlap * gauges[None, :]
    gauged_amplitudes = amplitudes / gauges
    permutation = torch.tensor([2, 0, 1])
    observed = contribution_gram_spectrum(
        gauged_overlap[permutation][:, permutation],
        gauged_amplitudes[permutation],
    )
    torch.testing.assert_close(observed, expected, atol=2e-13, rtol=2e-13)
    assert float(observed.sum()) == pytest.approx(1.0, abs=2e-13)


def test_leave_one_out_energies_match_direct_subspace_solves() -> None:
    pair_matrices = _pair_matrices()
    one_body, interaction = harmonic_pair_hamiltonian(6, kappa=0.2)
    overlap, hamiltonian = agp_hamiltonian_transition_matrices(
        pair_matrices, 2, one_body, interaction
    )
    observed = leave_one_out_energies(hamiltonian, overlap)
    for omitted in range(3):
        retained = torch.arange(3) != omitted
        expected = solve_generalized_hermitian(
            hamiltonian[retained][:, retained], overlap[retained][:, retained]
        ).energy
        torch.testing.assert_close(observed[omitted], expected)


def test_pruning_rule_ignores_raw_gauge_ill_conditioning() -> None:
    pair_matrices = _pair_matrices()
    one_body, interaction = harmonic_pair_hamiltonian(6, kappa=0.2)
    overlap, hamiltonian = agp_hamiltonian_transition_matrices(
        pair_matrices, 2, one_body, interaction
    )
    gauges = torch.tensor([1e4, 1e-4j, -2.0], dtype=overlap.dtype)
    gauged_overlap = gauges.conj()[:, None] * overlap * gauges[None, :]
    gauged_hamiltonian = (
        gauges.conj()[:, None] * hamiltonian * gauges[None, :]
    )
    assessment = assess_term_pruning(
        gauged_hamiltonian, gauged_overlap, condition_threshold=1e3
    )
    assert not assessment.should_prune
    assert assessment.candidate is None
    assert assessment.balanced_condition_number < 1e3


def test_pruning_rule_selects_a_duplicate_with_zero_energy_penalty() -> None:
    pair_matrices = _pair_matrices(terms=2)
    pair_matrices = torch.stack(
        (pair_matrices[0], pair_matrices[0], pair_matrices[1])
    )
    one_body, interaction = harmonic_pair_hamiltonian(6, kappa=0.2)
    overlap, hamiltonian = agp_hamiltonian_transition_matrices(
        pair_matrices, 2, one_body, interaction
    )
    assessment = assess_term_pruning(
        hamiltonian,
        overlap,
        condition_threshold=1e3,
        energy_tolerance=1e-9,
    )
    assert assessment.should_prune
    assert assessment.candidate in (0, 1)
    assert assessment.energy_penalty == pytest.approx(0.0, abs=2e-12)
    assert assessment.discarded_rank == 1

    full = solve_generalized_hermitian(hamiltonian, overlap)
    retained = torch.arange(3) != assessment.candidate
    reduced = solve_generalized_hermitian(
        hamiltonian[retained][:, retained], overlap[retained][:, retained]
    )
    coefficients = torch.stack(
        [agp_exterior_coefficients(matrix, 2) for matrix in pair_matrices]
    )
    full_state = full.amplitudes @ coefficients
    reduced_state = reduced.amplitudes @ coefficients[retained]
    fidelity = torch.abs(torch.vdot(full_state, reduced_state)) ** 2 / (
        torch.vdot(full_state, full_state).real
        * torch.vdot(reduced_state, reduced_state).real
    )
    torch.testing.assert_close(
        fidelity, torch.ones_like(fidelity), atol=2e-12, rtol=0
    )
