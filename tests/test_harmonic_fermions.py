import pytest
import torch

from femps.exterior import agp_tensor, apply_two_body_sum
from femps.hamiltonians import (
    agp_energy,
    antisymmetric_many_body_hamiltonian,
    antisymmetric_many_body_hamiltonian_dense_two_body,
    antisymmetric_two_particle_hamiltonian,
    exact_interacting_harmonic_fermion_energy,
    exact_interacting_pair_energy,
    exact_noninteracting_fermion_energy,
    harmonic_pair_hamiltonian,
)


def test_e1_noninteracting_pair_has_exact_energy_two() -> None:
    one_body, interaction = harmonic_pair_hamiltonian(6, kappa=0.0)
    pair_matrix = torch.zeros(6, 6, dtype=torch.complex128)
    pair_matrix[0, 1] = 1.0
    pair_matrix[1, 0] = -1.0
    assert exact_noninteracting_fermion_energy(2) == 2.0
    assert agp_energy(pair_matrix, 1, one_body, interaction).item() == pytest.approx(2.0)
    truth = torch.linalg.eigvalsh(
        antisymmetric_two_particle_hamiltonian(one_body, interaction)
    )[0]
    assert truth.item() == pytest.approx(2.0)


def test_e2_polynomial_energy_matches_full_particle_tensor() -> None:
    one_body, interaction = harmonic_pair_hamiltonian(7, kappa=0.35)
    generator = torch.Generator().manual_seed(101)
    raw = torch.randn(7, 7, generator=generator, dtype=torch.float64)
    pair_matrix = torch.complex(raw - raw.transpose(0, 1), torch.zeros_like(raw))
    polynomial = agp_energy(pair_matrix, 1, one_body, interaction)
    state = agp_tensor(pair_matrix, 1)
    identity = torch.eye(7, dtype=torch.complex128)
    full_one = torch.kron(one_body, identity) + torch.kron(identity, one_body)
    explicit_numerator = torch.vdot(
        state.reshape(-1),
        full_one @ state.reshape(-1)
        + apply_two_body_sum(state, interaction.dense()).reshape(-1),
    )
    explicit = (explicit_numerator / torch.vdot(state.reshape(-1), state.reshape(-1))).real
    torch.testing.assert_close(polynomial, explicit, atol=2e-11, rtol=2e-12)


def test_e2_truncated_truth_converges_to_separated_continuum_energy() -> None:
    continuum = exact_interacting_pair_energy(kappa=0.35)
    errors = []
    for order in (6, 10, 14):
        one_body, interaction = harmonic_pair_hamiltonian(order, kappa=0.35)
        truth = torch.linalg.eigvalsh(
            antisymmetric_two_particle_hamiltonian(one_body, interaction)
        )[0].item()
        errors.append(abs(truth - continuum))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 1e-8


def test_many_body_exterior_truth_reduces_to_two_particle_projection() -> None:
    one_body, interaction = harmonic_pair_hamiltonian(6, kappa=0.2)
    many_body = antisymmetric_many_body_hamiltonian(
        one_body, particles=2, two_body=interaction
    )
    projected = antisymmetric_two_particle_hamiltonian(one_body, interaction)
    torch.testing.assert_close(many_body, projected, atol=3e-14, rtol=3e-14)


def test_dense_two_body_slater_condon_matches_factorized_truth() -> None:
    one_body, interaction = harmonic_pair_hamiltonian(7, kappa=0.2)
    factorized = antisymmetric_many_body_hamiltonian(
        one_body, particles=4, two_body=interaction
    )
    dense = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, particles=4, two_body_tensor=interaction.dense()
    )
    torch.testing.assert_close(dense, factorized, atol=2e-13, rtol=2e-13)


def test_e3_four_fermion_slater_is_exact_agp_ground_state() -> None:
    dimension = 8
    particles = 4
    pairs = particles // 2
    one_body, interaction = harmonic_pair_hamiltonian(dimension, kappa=0.0)
    pair_matrix = torch.zeros(dimension, dimension, dtype=torch.complex128)
    pair_matrix[0, 1] = pair_matrix[2, 3] = 1
    pair_matrix[1, 0] = pair_matrix[3, 2] = -1
    energy = agp_energy(pair_matrix, pairs, one_body, interaction)
    truth = torch.linalg.eigvalsh(
        antisymmetric_many_body_hamiltonian(
            one_body, particles=particles, two_body=interaction
        )
    )[0]
    expected = exact_interacting_harmonic_fermion_energy(
        particles, kappa=0.0
    )
    assert expected == exact_noninteracting_fermion_energy(particles) == 8.0
    assert energy.item() == pytest.approx(expected, abs=2e-14)
    assert truth.item() == pytest.approx(expected, abs=2e-14)
