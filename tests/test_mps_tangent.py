import torch

from femps.algorithms.ordered_continuous_training import (
    random_uniform_functional_mps,
)
from femps.baselines.ordered_continuous_fourier import (
    ordered_continuous_fourier_hamiltonian_compressed_mpo,
)
from femps.benchmarks.mps_tangent import (
    left_gauge_physical_tangent_directions,
    mpo_energy_and_tangent_directional_derivatives,
)


def _small_state_and_mpo():
    mps = random_uniform_functional_mps(
        3, 3, 3, seed=1911, dtype=torch.float64
    )
    mpo, _ = ordered_continuous_fourier_hamiltonian_compressed_mpo(
        3,
        3,
        0.8,
        16,
        16,
        distance_basis="multiscale_odd_hermite",
        distance_scale_ratio=2.5,
        local_quadrature_order=64,
    )
    return mps, mpo


def test_left_gauge_tangent_directions_are_physical_unit_vectors() -> None:
    mps, _ = _small_state_and_mpo()
    canonical, directions = left_gauge_physical_tangent_directions(
        mps, directions_per_site=2, seed=1912
    )
    assert 2 <= len(directions) < 2 * canonical.N
    for direction in directions:
        assert abs(direction["normalized_physical_norm"] - 1) < 2e-12
        assert direction["state_overlap_absolute_value"] < 2e-12
        assert (
            direction["gauge_residual_before_physical_normalization"]
            < 2e-12
        )


def test_tangent_autograd_derivative_matches_centered_finite_difference() -> None:
    from latticetn.mps import MPS

    mps, mpo = _small_state_and_mpo()
    canonical, directions = left_gauge_physical_tangent_directions(
        mps, directions_per_site=1, seed=1913
    )
    energy, derivatives, _ = mpo_energy_and_tangent_directional_derivatives(
        canonical, mpo, directions
    )
    assert torch.isfinite(torch.tensor(energy))
    step = 2e-6
    for direction, observed in zip(directions, derivatives, strict=True):
        energies = []
        for sign in [-1, 1]:
            tensors = [tensor.detach().clone() for tensor in canonical.tensors]
            tensors[direction["site"]] = (
                tensors[direction["site"]]
                + sign * step * direction["tensor"]
            )
            shifted = MPS.from_tensors(
                tensors,
                dtype=torch.float64,
                device="cpu",
                requires_grad=False,
            )
            energies.append(float(shifted.energy_with_MPO(mpo)))
        expected = (energies[1] - energies[0]) / (2 * step)
        assert abs(observed - expected) < 2e-7
