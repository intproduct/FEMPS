import math

import pytest
import torch

from femps.ordered_continuous import (
    center_of_mass_distance_matrices,
    center_of_mass_distances_to_ordered,
    ordered_chamber_rescaling,
    ordered_harmonic_metric,
    ordered_kinetic_metric,
    ordered_to_center_of_mass_distances,
)


@pytest.mark.parametrize("particles", [1, 2, 3, 5])
def test_center_of_mass_distance_transform_is_unit_jacobian_bijection(
    particles: int,
) -> None:
    forward, inverse = center_of_mass_distance_matrices(particles)
    identity = torch.eye(particles, dtype=torch.float64)
    torch.testing.assert_close(forward @ inverse, identity, atol=2e-16, rtol=0)
    torch.testing.assert_close(inverse @ forward, identity, atol=2e-16, rtol=0)
    torch.testing.assert_close(
        torch.abs(torch.linalg.det(forward)),
        torch.ones((), dtype=torch.float64),
        atol=2e-16,
        rtol=0,
    )


def test_ordered_coordinate_roundtrip_and_positive_distances() -> None:
    coordinates = torch.tensor(
        [[-2.1, -0.4, 0.3, 1.8], [-1.0, -0.2, 2.0, 4.5]],
        dtype=torch.float64,
    )
    transformed = ordered_to_center_of_mass_distances(coordinates)
    assert bool(torch.all(transformed[:, 1:] > 0))
    torch.testing.assert_close(
        center_of_mass_distances_to_ordered(transformed),
        coordinates,
        atol=8e-16,
        rtol=8e-16,
    )
    with pytest.raises(ValueError, match="strictly ordered"):
        ordered_to_center_of_mass_distances(torch.tensor([0.0, 0.0]))
    with pytest.raises(ValueError, match="must be positive"):
        center_of_mass_distances_to_ordered(torch.tensor([0.0, -1.0]))


@pytest.mark.parametrize("particles", [2, 3, 6])
def test_kinetic_metric_has_decoupled_center_and_cartan_gap_block(
    particles: int,
) -> None:
    metric = ordered_kinetic_metric(particles)
    expected = torch.zeros_like(metric)
    expected[0, 0] = 1 / particles
    for distance in range(1, particles):
        expected[distance, distance] = 2
        if distance + 1 < particles:
            expected[distance, distance + 1] = -1
            expected[distance + 1, distance] = -1
    torch.testing.assert_close(metric, expected, atol=2e-16, rtol=0)


@pytest.mark.parametrize("particles", [2, 3, 6])
def test_harmonic_metric_matches_direct_coordinate_quadratic(
    particles: int,
) -> None:
    metric = ordered_harmonic_metric(particles)
    expected = torch.zeros_like(metric)
    expected[0, 0] = particles
    for left in range(1, particles):
        for right in range(1, particles):
            expected[left, right] = (
                min(left, right) * (particles - max(left, right)) / particles
            )
    torch.testing.assert_close(metric, expected, atol=3e-16, rtol=0)

    transformed = torch.linspace(-0.3, 1.2, particles, dtype=torch.float64)
    if particles > 1:
        transformed[1:] += 0.5
    coordinates = center_of_mass_distances_to_ordered(transformed)
    torch.testing.assert_close(
        0.5 * torch.dot(coordinates, coordinates),
        0.5 * transformed @ metric @ transformed,
        atol=2e-15,
        rtol=2e-15,
    )


def test_kinetic_metric_reproduces_direct_laplacian_chain_rule() -> None:
    particles = 4
    forward, _ = center_of_mass_distance_matrices(particles)
    generator = torch.Generator().manual_seed(161)
    raw = torch.randn(
        particles, particles, generator=generator, dtype=torch.float64
    )
    quadratic = 0.5 * (raw + raw.mT)
    coordinates = torch.randn(
        particles, generator=generator, dtype=torch.float64, requires_grad=True
    )

    def function(values: torch.Tensor) -> torch.Tensor:
        transformed = forward @ values
        return transformed @ quadratic @ transformed

    direct_hessian = torch.autograd.functional.hessian(function, coordinates)
    direct_laplacian = torch.trace(direct_hessian)
    expected = 2 * torch.sum(ordered_kinetic_metric(particles) * quadratic)
    torch.testing.assert_close(direct_laplacian, expected, atol=2e-14, rtol=2e-14)


def test_ordered_chamber_rescaling_is_sqrt_factorial() -> None:
    assert ordered_chamber_rescaling(4) == math.sqrt(24)
