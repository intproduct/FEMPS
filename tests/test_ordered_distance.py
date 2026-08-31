import math

import torch

from femps.ordered_distance import (
    gap_configuration_to_ordered,
    gap_configurations,
    gap_hamiltonian,
    gap_tensor_to_ordered_values,
    ordered_configuration_to_gaps,
    ordered_hamiltonian_to_gap_basis,
    ordered_values_to_gap_tensor,
)
from femps.ordered_sector import (
    finite_difference_harmonic_hamiltonian,
    ordered_configurations,
    ordered_sector_hamiltonian,
)


def test_gap_coordinates_are_a_bijection_with_the_ordered_sector() -> None:
    for grid_points, particles in ((5, 2), (7, 3), (8, 4)):
        ordered = ordered_configurations(grid_points, particles)
        gaps = gap_configurations(grid_points, particles)
        assert len(gaps) == len(ordered) == math.comb(grid_points, particles)
        assert {
            gap_configuration_to_ordered(configuration, grid_points)
            for configuration in gaps
        } == set(ordered)
        assert all(
            gap_configuration_to_ordered(
                ordered_configuration_to_gaps(configuration, grid_points),
                grid_points,
            )
            == configuration
            for configuration in ordered
        )


def test_gap_tensor_preserves_ordered_values_and_norm() -> None:
    generator = torch.Generator().manual_seed(3)
    values = torch.randn(35, generator=generator, dtype=torch.float64)
    tensor = ordered_values_to_gap_tensor(values, 7, 3)
    assert tensor.shape == (5, 5, 5, 5)
    torch.testing.assert_close(
        gap_tensor_to_ordered_values(tensor, 7, 3), values, atol=0, rtol=0
    )
    torch.testing.assert_close(torch.vdot(tensor.flatten(), tensor.flatten()), torch.vdot(values, values))


def test_gap_kinetic_harmonic_and_soft_coulomb_match_ordered_truth() -> None:
    grid_points = 8
    particles = 4
    spacing = 0.7
    grid, one_body = finite_difference_harmonic_hamiltonian(
        grid_points, spacing
    )
    pair_potential = 1 / torch.sqrt(
        (grid[:, None] - grid[None, :]) ** 2 + 1
    )
    ordered = ordered_sector_hamiltonian(
        one_body, particles, pair_potential=pair_potential
    )
    expected = ordered_hamiltonian_to_gap_basis(
        ordered, grid_points, particles
    )
    observed = gap_hamiltonian(
        grid_points,
        particles,
        spacing,
        soft_coulomb=True,
        softening=1.0,
    )
    torch.testing.assert_close(observed, expected, atol=3e-15, rtol=3e-15)
