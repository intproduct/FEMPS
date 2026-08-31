import pytest
import torch

from femps.exterior import (
    agp_tensor,
    antisymmetry_residual,
    bivector_decomposition_length,
    materialize_femps_matrix,
    materialize_femps_as_agp_path_sum,
    materialize_femps_paths,
    normalized_slater_from_minors,
    slater_sum_cores,
    slater_as_agp_pair_matrix,
    wedge_tensors,
)


def _random_complex(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imag = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imag)


def test_wedge_is_associative_in_normalized_tensor_convention() -> None:
    u, v, w = _random_complex((3, 6), 4)
    left = wedge_tensors(wedge_tensors(u, v), w)
    right = wedge_tensors(u, wedge_tensors(v, w))
    torch.testing.assert_close(left, right, atol=2e-13, rtol=2e-13)
    assert antisymmetry_residual(left).item() < 1e-14


def test_chi_one_femps_is_exactly_one_decomposable_wedge() -> None:
    orbitals = _random_complex((6, 4), 5)
    cores = [orbitals[:, site].reshape(1, 6, 1) for site in range(4)]
    observed = materialize_femps_matrix(cores)
    expected = normalized_slater_from_minors(orbitals)
    torch.testing.assert_close(observed, expected, atol=3e-13, rtol=3e-13)


def test_every_even_slater_is_one_fixed_number_agp() -> None:
    orbitals = _random_complex((6, 4), 7)
    pair_matrix = slater_as_agp_pair_matrix(orbitals)
    torch.testing.assert_close(
        agp_tensor(pair_matrix, 2),
        normalized_slater_from_minors(orbitals),
        atol=3e-12,
        rtol=3e-12,
    )


@pytest.mark.parametrize(
    "bonds",
    [
        (1, 2, 1),
        (1, 2, 3, 1),
        (1, 2, 3, 2, 1),
    ],
)
def test_matrix_wedge_and_path_enumeration_agree(bonds: tuple[int, ...]) -> None:
    particles = len(bonds) - 1
    cores = [
        _random_complex((bonds[site], 5, bonds[site + 1]), 11 + site)
        for site in range(particles)
    ]
    by_matrix = materialize_femps_matrix(cores)
    by_paths = materialize_femps_paths(cores)
    torch.testing.assert_close(by_matrix, by_paths, atol=2e-12, rtol=2e-12)
    assert antisymmetry_residual(by_matrix).item() < 1e-14


def test_even_femps_equals_its_pathwise_lc_agp_expansion() -> None:
    bonds = (1, 2, 3, 2, 1)
    cores = [
        _random_complex((bonds[site], 5, bonds[site + 1]), 71 + site)
        for site in range(4)
    ]
    torch.testing.assert_close(
        materialize_femps_as_agp_path_sum(cores),
        materialize_femps_matrix(cores),
        atol=4e-11,
        rtol=4e-11,
    )


def test_finite_slater_sum_has_diagonal_path_embedding() -> None:
    orbitals = _random_complex((3, 5, 3), 21)
    weights = _random_complex((3,), 22)
    cores = slater_sum_cores(orbitals, weights)
    observed = materialize_femps_matrix(cores)
    expected = sum(
        weights[term] * normalized_slater_from_minors(orbitals[term])
        for term in range(orbitals.shape[0])
    )
    torch.testing.assert_close(observed, expected, atol=2e-12, rtol=2e-12)


def test_ordinary_mps_gauge_action_leaves_femps_state_invariant() -> None:
    first = _random_complex((1, 5, 2), 31)
    second = _random_complex((2, 5, 1), 32)
    gauge = _random_complex((2, 2), 33) + 2.0 * torch.eye(2)
    inverse = torch.linalg.inv(gauge)
    transformed_first = torch.einsum("ldr,rs->lds", first, gauge)
    transformed_second = torch.einsum("lr,rdq->ldq", inverse, second)
    torch.testing.assert_close(
        materialize_femps_matrix([first, second]),
        materialize_femps_matrix([transformed_first, transformed_second]),
        atol=2e-12,
        rtol=2e-12,
    )


def test_bivector_minimal_decomposable_length_is_half_skew_rank() -> None:
    u, v = _random_complex((2, 6), 41)
    single = wedge_tensors(u, v)
    assert bivector_decomposition_length(single) == 1

    raw = _random_complex((6, 6), 42)
    generic = raw - raw.transpose(0, 1)
    assert bivector_decomposition_length(generic) == 3
