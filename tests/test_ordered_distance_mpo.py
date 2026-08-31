import pytest
import torch

pytestmark = pytest.mark.integration
pytest.importorskip("latticetn")

from femps.baselines.ordered_distance_mpo import (
    compress_mpo,
    gap_charge_projector_mpo,
    gap_kinetic_harmonic_mpo,
    gap_soft_coulomb_hamiltonian_mpo,
    gap_soft_coulomb_mpo,
    ordered_values_to_gap_mps,
)
from femps.ordered_distance import (
    gap_configurations,
    gap_hamiltonian,
    ordered_values_to_gap_values,
    ordered_hamiltonian_to_gap_basis,
)
from femps.ordered_sector import (
    finite_difference_harmonic_hamiltonian,
    ordered_sector_hamiltonian,
)


def _flat_gap_indices(grid_points: int, particles: int) -> torch.Tensor:
    local_dimension = grid_points - particles + 1
    sites = particles + 1
    return torch.tensor(
        [
            sum(value * local_dimension ** (sites - 1 - site) for site, value in enumerate(gaps))
            for gaps in gap_configurations(grid_points, particles)
        ],
        dtype=torch.long,
    )


def _flat_truncated_gap_indices(
    grid_points: int, particles: int, gap_cutoff: int
) -> torch.Tensor:
    sites = particles + 1
    local_dimension = gap_cutoff + 1
    return torch.tensor(
        [
            sum(value * local_dimension ** (sites - 1 - site) for site, value in enumerate(gaps))
            for gaps in gap_configurations(
                grid_points, particles, gap_cutoff=gap_cutoff
            )
        ],
        dtype=torch.long,
    )


def test_native_mpo_matches_exact_gap_kinetic_and_harmonic_sector() -> None:
    grid_points = 6
    particles = 3
    spacing = 0.7
    mpo = gap_kinetic_harmonic_mpo(grid_points, particles, spacing)
    dense = mpo.to_dense()
    indices = _flat_gap_indices(grid_points, particles)
    observed = dense[indices][:, indices]
    expected = gap_hamiltonian(grid_points, particles, spacing)
    torch.testing.assert_close(observed, expected, atol=3e-14, rtol=3e-14)
    outside = torch.ones(dense.shape[0], dtype=torch.bool)
    outside[indices] = False
    torch.testing.assert_close(
        dense[indices][:, outside], torch.zeros_like(dense[indices][:, outside])
    )


def test_gap_charge_projector_selects_only_the_finite_box_sector() -> None:
    grid_points = 6
    particles = 3
    projector = gap_charge_projector_mpo(grid_points, particles)
    diagonal = torch.diag(projector.to_dense())
    indices = _flat_gap_indices(grid_points, particles)
    expected = torch.zeros_like(diagonal)
    expected[indices] = 1
    torch.testing.assert_close(diagonal, expected, atol=0, rtol=0)
    assert projector.tensors[1].shape[0] == grid_points - particles + 1


def test_gap_ground_state_uses_only_native_latticetn_mps_mpo_contraction() -> None:
    grid_points = 8
    particles = 4
    spacing = 0.7
    _, one_body = finite_difference_harmonic_hamiltonian(grid_points, spacing)
    ordered_hamiltonian = ordered_sector_hamiltonian(one_body, particles)
    eigenvalues, eigenvectors = torch.linalg.eigh(ordered_hamiltonian)
    mps, ranks, discarded = ordered_values_to_gap_mps(
        eigenvectors[:, 0], grid_points, particles
    )
    mpo = gap_kinetic_harmonic_mpo(grid_points, particles, spacing)
    energy = mps.energy_with_MPO(mpo)
    torch.testing.assert_close(energy, eigenvalues[0], atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(
        mps.norm_sq(), torch.ones((), dtype=torch.float64), atol=3e-13, rtol=3e-13
    )
    assert discarded < 1e-13
    assert max(ranks) <= len(gap_configurations(grid_points, particles))
    energy.backward()
    assert all(core.grad is not None and torch.isfinite(core.grad).all() for core in mps.tensors)


def test_interval_count_soft_coulomb_mpo_is_exact_on_fixed_charge_sector() -> None:
    grid_points = 6
    particles = 3
    spacing = 0.7
    mpo = gap_soft_coulomb_mpo(grid_points, particles, spacing)
    dense = mpo.to_dense()
    indices = _flat_gap_indices(grid_points, particles)
    observed = dense[indices][:, indices]
    full = gap_hamiltonian(
        grid_points, particles, spacing, soft_coulomb=True
    )
    noninteracting = gap_hamiltonian(grid_points, particles, spacing)
    expected = full - noninteracting
    torch.testing.assert_close(observed, expected, atol=3e-15, rtol=3e-15)


def test_soft_coulomb_ground_energy_is_native_mps_mpo() -> None:
    grid_points = 8
    particles = 4
    spacing = 0.7
    grid, one_body = finite_difference_harmonic_hamiltonian(grid_points, spacing)
    pair_potential = 1 / torch.sqrt((grid[:, None] - grid[None, :]) ** 2 + 1)
    ordered_hamiltonian = ordered_sector_hamiltonian(
        one_body, particles, pair_potential=pair_potential
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(ordered_hamiltonian)
    mps, _, discarded = ordered_values_to_gap_mps(
        eigenvectors[:, 0], grid_points, particles
    )
    mpo = gap_soft_coulomb_hamiltonian_mpo(
        grid_points, particles, spacing
    )
    energy = mps.energy_with_MPO(mpo)
    torch.testing.assert_close(energy, eigenvalues[0], atol=5e-12, rtol=5e-12)
    assert discarded < 1e-13
    energy.backward()
    assert all(core.grad is not None and torch.isfinite(core.grad).all() for core in mps.tensors)


def test_gap_cutoff_is_an_independent_exact_projected_basis_control() -> None:
    grid_points = 8
    particles = 4
    spacing = 0.7
    gap_cutoff = 2
    mpo = gap_soft_coulomb_hamiltonian_mpo(
        grid_points, particles, spacing, gap_cutoff=gap_cutoff
    )
    dense = mpo.to_dense()
    indices = _flat_truncated_gap_indices(
        grid_points, particles, gap_cutoff
    )
    observed = dense[indices][:, indices]
    expected = gap_hamiltonian(
        grid_points,
        particles,
        spacing,
        gap_cutoff=gap_cutoff,
        soft_coulomb=True,
    )
    torch.testing.assert_close(observed, expected, atol=3e-14, rtol=3e-14)


def test_mpo_bond_compression_has_independently_measured_operator_error() -> None:
    grid_points = 6
    particles = 3
    spacing = 0.7
    exact = gap_soft_coulomb_hamiltonian_mpo(
        grid_points, particles, spacing
    )
    compressed, ranks, discarded = compress_mpo(exact, 6)
    indices = _flat_gap_indices(grid_points, particles)
    exact_sector = exact.to_dense()[indices][:, indices]
    compressed_sector = compressed.to_dense()[indices][:, indices]
    assert max(ranks) <= 6
    assert discarded > 0
    assert torch.linalg.matrix_norm(compressed_sector - exact_sector) > 0
    exact_recompressed, exact_ranks, exact_discarded = compress_mpo(exact, 64)
    torch.testing.assert_close(
        exact_recompressed.to_dense()[indices][:, indices],
        exact_sector,
        atol=2e-13,
        rtol=2e-13,
    )
    assert max(exact_ranks) <= 64
    assert exact_discarded < 1e-13
