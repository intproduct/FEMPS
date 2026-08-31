import torch

from femps.baselines.coupled_oscillators import (
    dense_truncated_hamiltonian,
    exact_ground_energy,
    normal_mode_frequencies,
)


def test_exact_energy_is_invariant_under_coupling_sign_for_open_chain():
    positive = exact_ground_energy(6, gamma=0.3)
    negative = exact_ground_energy(6, gamma=-0.3)
    assert abs(positive - negative) < 1e-14


def test_normal_modes_reject_unbounded_quadratic_form():
    try:
        normal_mode_frequencies(16, gamma=0.7)
    except ValueError as exc:
        assert "not positive definite" in str(exc)
    else:
        raise AssertionError("unstable oscillator chain was accepted")


def test_small_dense_truncation_converges_to_continuum_energy():
    reference = exact_ground_energy(2, gamma=-0.5)
    hamiltonian = dense_truncated_hamiltonian(2, 10, gamma=-0.5)
    truncated = float(torch.linalg.eigvalsh(hamiltonian)[0])
    assert abs(truncated - reference) < 2e-8

