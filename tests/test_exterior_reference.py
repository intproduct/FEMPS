import math

import pytest
import torch

from femps.exterior import (
    alternating_projection,
    antisymmetry_residual,
    best_rank_error,
    normalized_slater_from_antisymmetrizer,
    normalized_slater_from_minors,
    particle_schmidt_spectrum,
    particle_tt_ranks,
    particle_unfolding,
    slater_flat_spectrum,
)


def _orthonormal_orbitals(dimension: int, particles: int, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(dimension, particles, generator=generator, dtype=torch.float64)
    imag = torch.randn(dimension, particles, generator=generator, dtype=torch.float64)
    orbitals, _ = torch.linalg.qr(torch.complex(real, imag), mode="reduced")
    return orbitals


def test_two_independent_slater_materializations_agree() -> None:
    orbitals = _orthonormal_orbitals(5, 3)
    by_projection = normalized_slater_from_antisymmetrizer(orbitals)
    by_minors = normalized_slater_from_minors(orbitals)

    torch.testing.assert_close(by_projection, by_minors, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(
        torch.linalg.vector_norm(by_minors),
        torch.tensor(1.0, dtype=by_minors.real.dtype),
    )
    assert antisymmetry_residual(by_minors).item() < 1e-14


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_slater_particle_spectrum_is_flat(cut: int) -> None:
    particles = 4
    state = normalized_slater_from_minors(_orthonormal_orbitals(6, particles))
    observed = particle_schmidt_spectrum(state, cut)
    expected = slater_flat_spectrum(particles, cut, dtype=observed.dtype)

    torch.testing.assert_close(observed[: expected.numel()], expected, atol=2e-14, rtol=2e-14)
    assert torch.count_nonzero(observed[expected.numel() :] > 1e-13) == 0
    assert particle_tt_ranks(state)[cut - 1] == math.comb(particles, cut)


def test_slater_best_rank_error_matches_closed_form() -> None:
    particles, cut = 4, 2
    state = normalized_slater_from_minors(_orthonormal_orbitals(6, particles))
    spectrum = particle_schmidt_spectrum(state, cut)
    multiplicity = math.comb(particles, cut)
    for rank in range(multiplicity + 1):
        expected = math.sqrt(1.0 - rank / multiplicity)
        assert best_rank_error(spectrum, rank).item() == pytest.approx(expected, abs=2e-14)


def test_nonzero_alternating_tensor_obeys_particle_rank_floor() -> None:
    generator = torch.Generator().manual_seed(17)
    raw = torch.randn((5,) * 3, generator=generator, dtype=torch.float64)
    alternating = alternating_projection(raw)

    assert torch.linalg.vector_norm(alternating) > 0
    assert particle_tt_ranks(alternating) == (5, 5)
    assert all(rank >= math.comb(3, cut) for cut, rank in enumerate((5, 5), start=1))


def test_ordinary_low_rank_truncation_breaks_full_antisymmetry() -> None:
    state = normalized_slater_from_minors(_orthonormal_orbitals(6, 4))
    unfolding = particle_unfolding(state, 2)
    left, values, right_h = torch.linalg.svd(unfolding, full_matrices=False)
    truncated = (left[:, :5] * values[:5]) @ right_h[:5, :]
    truncated = truncated.reshape(state.shape)

    assert torch.linalg.matrix_rank(particle_unfolding(truncated, 2)).item() == 5
    assert antisymmetry_residual(truncated).item() > 1e-3


def test_reference_validation_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        alternating_projection(torch.ones(2, 3))
    with pytest.raises(ValueError):
        slater_flat_spectrum(4, 0)
