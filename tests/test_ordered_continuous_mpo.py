import math

import pytest
import torch

pytestmark = pytest.mark.integration
pytest.importorskip("latticetn")

from femps.baselines.ordered_continuous_mpo import (
    ordered_continuous_local_operators,
    ordered_continuous_noninteracting_mpo,
    ordered_continuous_soft_coulomb_hamiltonian_mpo,
)
from femps.baselines.ordered_functional_mps import particle_tensor_to_mps_tensors
from femps.hamiltonians.soft_coulomb import (
    soft_coulomb_two_fermion_relative_grid_energy,
)


def test_n2_continuous_mpo_matches_independent_separable_hamiltonian() -> None:
    particles = 2
    order = 7
    distance_length = 8.0
    operators = ordered_continuous_local_operators(
        particles, order, distance_length
    )
    identity = torch.eye(order, dtype=torch.float64)
    center = (
        0.25 * operators[0]["negative_second_derivative"]
        + operators[0]["position_squared"]
    )
    relative = (
        operators[1]["negative_second_derivative"]
        + 0.25 * operators[1]["position_squared"]
    )
    expected = torch.kron(center, identity) + torch.kron(identity, relative)
    observed = ordered_continuous_noninteracting_mpo(
        particles, order, distance_length
    ).to_dense()
    torch.testing.assert_close(observed, expected, atol=3e-15, rtol=3e-15)


def test_n3_mixed_derivative_and_harmonic_mpo_is_hermitian() -> None:
    mpo = ordered_continuous_noninteracting_mpo(3, 5, 7.0)
    dense = mpo.to_dense()
    torch.testing.assert_close(dense.mT, dense, atol=2e-14, rtol=2e-14)
    assert max(max(tensor.shape[:2]) for tensor in mpo.tensors) <= 8


def test_n2_functional_basis_converges_to_exact_fermion_energy() -> None:
    errors = []
    for order, distance_length in ((6, 6.0), (10, 8.0), (14, 10.0)):
        dense = ordered_continuous_noninteracting_mpo(
            2, order, distance_length
        ).to_dense()
        energy = torch.linalg.eigvalsh(dense)[0]
        errors.append(float(torch.abs(energy - 2.0)))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[-1] < 2e-6
    assert math.isfinite(sum(errors))


def test_n3_mixed_derivative_basis_converges_to_exact_fermion_energy() -> None:
    errors = []
    for order in (4, 6, 8):
        dense = ordered_continuous_noninteracting_mpo(
            3, order, 6.0
        ).to_dense()
        energy = torch.linalg.eigvalsh(dense)[0]
        errors.append(float(energy - 4.5))
    assert 0 < errors[2] < errors[1] < errors[0]
    assert errors[-1] < 7e-3


def test_odd_hermite_half_line_basis_improves_noninteracting_n3_convergence() -> None:
    sine = ordered_continuous_noninteracting_mpo(3, 6, 6.0).to_dense()
    odd = ordered_continuous_noninteracting_mpo(
        3, 6, 1.0, distance_basis="odd_hermite"
    ).to_dense()
    sine_error = float(torch.linalg.eigvalsh(sine)[0] - 4.5)
    odd_error = abs(float(torch.linalg.eigvalsh(odd)[0] - 4.5))
    assert odd_error < sine_error


def test_soft_coulomb_rejects_unbounded_basis_until_interaction_is_controlled() -> None:
    with pytest.raises(ValueError, match="requires dirichlet_sine"):
        ordered_continuous_soft_coulomb_hamiltonian_mpo(
            2, 4, 1.0, 8, distance_basis="odd_hermite"
        )


def test_n3_continuous_truth_state_uses_native_mps_mpo_energy_and_ad() -> None:
    from latticetn.mps import MPS

    particles = 3
    order = 6
    mpo = ordered_continuous_noninteracting_mpo(particles, order, 6.0)
    eigenvalues, eigenvectors = torch.linalg.eigh(mpo.to_dense())
    cores, ranks, discarded = particle_tensor_to_mps_tensors(
        eigenvectors[:, 0].reshape((order,) * particles)
    )
    mps = MPS.from_tensors(
        cores,
        dtype=torch.float64,
        device="cpu",
        requires_grad=True,
    )
    energy = mps.energy_with_MPO(mpo)
    torch.testing.assert_close(energy, eigenvalues[0], atol=3e-13, rtol=3e-13)
    assert ranks == (1, 6)
    assert discarded < 1e-13
    energy.backward()
    assert all(
        core.grad is not None and bool(torch.isfinite(core.grad).all())
        for core in mps.tensors
    )


def test_zero_coupling_soft_coulomb_builder_reduces_to_noninteracting_mpo() -> None:
    observed = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        2, 5, 7.0, 12, coupling=0
    ).to_dense()
    expected = ordered_continuous_noninteracting_mpo(2, 5, 7.0).to_dense()
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)


def test_n2_interacting_functional_basis_matches_independent_grid_extrapolation() -> None:
    coarse = soft_coulomb_two_fermion_relative_grid_energy(
        intervals=360, half_width=8.0
    )
    fine = soft_coulomb_two_fermion_relative_grid_energy(
        intervals=720, half_width=8.0
    )
    richardson = (4 * fine - coarse) / 3
    hamiltonian = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        2,
        12,
        9.0,
        20,
        interaction_quadrature_order=160,
    ).to_dense()
    energy = float(torch.linalg.eigvalsh(hamiltonian)[0])
    assert abs(energy - richardson) < 2e-6
