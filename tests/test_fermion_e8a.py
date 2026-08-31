import pytest
import torch

from femps.basis import harmonic_hamiltonian
from femps.hamiltonians import agp_energy
from scripts.reproduce_fermion_e8a import occupied_pair_matrix


def test_eight_fermion_occupied_pair_matrix_has_exact_energy() -> None:
    pair_matrix = occupied_pair_matrix(10, 8)
    assert torch.allclose(pair_matrix, -pair_matrix.transpose(0, 1))
    energy = agp_energy(
        pair_matrix,
        4,
        harmonic_hamiltonian(10, dtype=torch.complex128),
    )
    assert float(energy) == pytest.approx(32.0, abs=1e-12)


@pytest.mark.parametrize("basis_order,particles", [(8, 7), (7, 8)])
def test_occupied_pair_matrix_rejects_invalid_sector(
    basis_order: int, particles: int
) -> None:
    with pytest.raises(ValueError):
        occupied_pair_matrix(basis_order, particles)
