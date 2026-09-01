import math

import pytest
import torch

from femps.exterior import (
    agp_exterior_coefficients,
    matrix_pair_exterior_coefficients,
    triangular_pair_lc_agp_decomposition,
    triangular_pair_lc_agp_term_bound,
)


def _random_complex(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imaginary)


def _random_skew(dimension: int, seed: int) -> torch.Tensor:
    raw = _random_complex((dimension, dimension), seed)
    return raw - raw.transpose(0, 1)


def _triangular_pair_matrices(dimension: int, seed: int) -> torch.Tensor:
    pair_matrices = torch.zeros(
        dimension, dimension, 2, 2, dtype=torch.complex128
    )
    pair_matrices[:, :, 0, 0] = _random_skew(dimension, seed)
    pair_matrices[:, :, 0, 1] = _random_skew(dimension, seed + 1)
    pair_matrices[:, :, 1, 1] = _random_skew(dimension, seed + 2)
    return pair_matrices


@pytest.mark.parametrize("pairs", [1, 2, 3])
def test_upper_triangular_pair_power_collapses_exactly_to_lc_agp(
    pairs: int,
) -> None:
    dimension = 2 * pairs + 2
    pair_matrices = _triangular_pair_matrices(dimension, 200 + pairs)
    left = _random_complex((2,), 210 + pairs)
    right = _random_complex((2,), 220 + pairs)
    lc_pairs, amplitudes = triangular_pair_lc_agp_decomposition(
        pair_matrices, pairs, left, right
    )
    observed = matrix_pair_exterior_coefficients(
        pair_matrices, pairs, left, right
    )
    expected = sum(
        amplitude * agp_exterior_coefficients(pair_matrix, pairs)
        for pair_matrix, amplitude in zip(lc_pairs, amplitudes, strict=True)
    )
    assert lc_pairs.shape[0] <= triangular_pair_lc_agp_term_bound(pairs)
    assert triangular_pair_lc_agp_term_bound(pairs) == (
        math.comb(pairs + 2, 2) + 2
    )
    torch.testing.assert_close(observed, expected, atol=2e-9, rtol=2e-10)


def test_collapse_holds_for_genuinely_noncommuting_coefficients() -> None:
    pair_matrices = _triangular_pair_matrices(6, 230)
    first = pair_matrices[0, 1]
    second = pair_matrices[2, 3]
    assert torch.linalg.matrix_norm(first @ second - second @ first) > 1e-3
    left = _random_complex((2,), 231)
    right = _random_complex((2,), 232)
    lc_pairs, amplitudes = triangular_pair_lc_agp_decomposition(
        pair_matrices, 2, left, right
    )
    expected = sum(
        amplitude * agp_exterior_coefficients(pair_matrix, 2)
        for pair_matrix, amplitude in zip(lc_pairs, amplitudes, strict=True)
    )
    torch.testing.assert_close(
        matrix_pair_exterior_coefficients(pair_matrices, 2, left, right),
        expected,
        atol=3e-10,
        rtol=3e-11,
    )


def test_lc_agp_collapse_preserves_restricted_reverse_mode_gradient() -> None:
    pair_matrices = _triangular_pair_matrices(6, 240).requires_grad_()
    left = _random_complex((2,), 241)
    right = _random_complex((2,), 242)
    direct = matrix_pair_exterior_coefficients(
        pair_matrices, 2, left, right
    )
    direct_loss = torch.vdot(direct, direct).real
    direct_gradient = torch.autograd.grad(
        direct_loss, pair_matrices, retain_graph=True
    )[0]
    lc_pairs, amplitudes = triangular_pair_lc_agp_decomposition(
        pair_matrices, 2, left, right
    )
    collapsed = sum(
        amplitude * agp_exterior_coefficients(pair_matrix, 2)
        for pair_matrix, amplitude in zip(lc_pairs, amplitudes, strict=True)
    )
    collapsed_loss = torch.vdot(collapsed, collapsed).real
    collapsed_gradient = torch.autograd.grad(collapsed_loss, pair_matrices)[0]
    torch.testing.assert_close(collapsed_loss, direct_loss, atol=3e-9, rtol=3e-11)
    physical_upper = torch.triu_indices(6, 6, offset=1)
    for virtual_row, virtual_column in [(0, 0), (0, 1), (1, 1)]:
        torch.testing.assert_close(
            collapsed_gradient[
                physical_upper[0],
                physical_upper[1],
                virtual_row,
                virtual_column,
            ],
            direct_gradient[
                physical_upper[0],
                physical_upper[1],
                virtual_row,
                virtual_column,
            ],
            atol=2e-8,
            rtol=2e-9,
        )


def test_collapse_rejects_a_lower_triangular_coefficient() -> None:
    pair_matrices = _triangular_pair_matrices(6, 250)
    pair_matrices[0, 1, 1, 0] = 1
    pair_matrices[1, 0, 1, 0] = -1
    boundary = torch.ones(2, dtype=torch.complex128)
    with pytest.raises(ValueError, match="upper triangular"):
        triangular_pair_lc_agp_decomposition(
            pair_matrices, 2, boundary, boundary
        )
