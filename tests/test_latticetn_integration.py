import pytest
import torch

pytestmark = pytest.mark.integration

pytest.importorskip("latticetn")

from femps.baselines.coupled_oscillators import (
    dense_truncated_hamiltonian,
    functional_mps_energy,
)
from femps.baselines.functional_mps import random_functional_mps


def test_functional_initializer_uses_local_basis_dimension_for_bond_cap():
    mps = random_functional_mps(4, 3, 9, seed=2)
    assert [tuple(t.shape) for t in mps.tensors] == [
        (1, 3, 3),
        (3, 3, 9),
        (9, 3, 3),
        (3, 3, 1),
    ]


def test_native_functional_energy_matches_dense_rayleigh_quotient():
    mps = random_functional_mps(3, 3, 5, seed=3)
    native = functional_mps_energy(mps, gamma=-0.2)
    state = mps.to_dense()
    hamiltonian = dense_truncated_hamiltonian(
        3, 3, gamma=-0.2, dtype=torch.complex128
    )
    dense = ((state.conj() @ (hamiltonian @ state)) / (state.conj() @ state)).real
    torch.testing.assert_close(native, dense, rtol=1e-11, atol=1e-11)
    native.backward()
    assert all(tensor.grad is not None for tensor in mps.tensors)

