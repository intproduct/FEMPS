import pytest
import torch

pytestmark = pytest.mark.integration
pytest.importorskip("latticetn")

from femps.baselines.ordered_functional_mps import (
    ordered_sector_dense_energy_from_mps,
    ordered_sector_functional_mps,
    ordered_values_from_mps,
    ordered_values_to_particle_tensor,
)
from femps.exterior import particle_tt_ranks
from femps.ordered_sector import (
    extend_from_ordered_sector,
    finite_difference_harmonic_hamiltonian,
    ordered_sector_hamiltonian,
)


def test_ordered_values_have_exact_latticetn_mps_and_native_norm() -> None:
    generator = torch.Generator().manual_seed(1)
    values = torch.randn(10, generator=generator, dtype=torch.float64)
    mps, ranks, discarded = ordered_sector_functional_mps(values, 5, 3)
    torch.testing.assert_close(ordered_values_from_mps(mps), values, atol=2e-14, rtol=2e-14)
    tensor = ordered_values_to_particle_tensor(values, 5, 3)
    torch.testing.assert_close(
        mps.to_dense().reshape(tensor.shape), tensor, atol=2e-14, rtol=2e-14
    )
    torch.testing.assert_close(mps.norm_sq(), torch.vdot(values, values), atol=2e-13, rtol=2e-13)
    assert ranks == particle_tt_ranks(tensor)
    assert discarded < 1e-14


def test_ordered_dense_energy_uses_latticetn_ad_parameters() -> None:
    _, one_body = finite_difference_harmonic_hamiltonian(6, 0.7)
    hamiltonian = ordered_sector_hamiltonian(one_body, 3)
    values = torch.linalg.eigh(hamiltonian).eigenvectors[:, 0]
    mps, _, _ = ordered_sector_functional_mps(values, 6, 3)
    observed = ordered_sector_dense_energy_from_mps(mps, hamiltonian)
    expected = torch.linalg.eigvalsh(hamiltonian)[0]
    torch.testing.assert_close(observed, expected, atol=2e-12, rtol=2e-12)
    observed.backward()
    assert all(core.grad is not None and torch.isfinite(core.grad).all() for core in mps.tensors)


def test_n4_soft_coulomb_ordered_tensor_removes_exchange_rank_multiplicity() -> None:
    dimension = 8
    particles = 4
    grid, one_body = finite_difference_harmonic_hamiltonian(dimension, 0.7)
    pair_potential = 1 / torch.sqrt((grid[:, None] - grid[None, :]) ** 2 + 1)
    hamiltonian = ordered_sector_hamiltonian(
        one_body, particles, pair_potential=pair_potential
    )
    values = torch.linalg.eigh(hamiltonian).eigenvectors[:, 0]
    ordered_tensor = ordered_values_to_particle_tensor(values, dimension, particles)
    antisymmetric_tensor = extend_from_ordered_sector(values, dimension, particles)
    assert particle_tt_ranks(ordered_tensor) == (5, 9, 5)
    assert particle_tt_ranks(antisymmetric_tensor) == (8, 28, 8)
