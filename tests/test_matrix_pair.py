import itertools

import torch

from femps.exterior import (
    agp_exterior_coefficients,
    fermionic_correlation_multiplicity,
    matrix_pair_exterior_coefficients,
    matrix_pair_exterior_matrices,
    matrix_pair_femps_cores,
    matrix_pair_n4_anticommutator,
    matrix_pair_norm,
    matrix_pair_one_body_expectation,
    matrix_pair_tensor,
    one_body_density_exterior_coefficients,
    apply_one_body_sum,
    tagged_cayley_expected_amplitude,
    tagged_cayley_pair_data,
    femps_exterior_coefficients,
)


def _random_complex(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imag = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imag)


def _random_skew_matrix_pairs(dimension: int, bond: int, seed: int) -> torch.Tensor:
    raw = _random_complex((dimension, dimension, bond, bond), seed)
    return raw - raw.transpose(0, 1)


def test_virtual_dimension_one_is_scalar_agp() -> None:
    pair_matrix = _random_skew_matrix_pairs(7, 1, 10)[:, :, 0, 0]
    pair_matrices = pair_matrix[:, :, None, None]
    boundary = torch.ones(1, dtype=pair_matrix.dtype)
    torch.testing.assert_close(
        matrix_pair_exterior_coefficients(pair_matrices, 3, boundary, boundary),
        agp_exterior_coefficients(pair_matrix, 3),
        atol=2e-11,
        rtol=2e-11,
    )


def test_n4_recurrence_equals_anticommutator_formula() -> None:
    pair_matrices = _random_skew_matrix_pairs(6, 2, 20)
    torch.testing.assert_close(
        matrix_pair_exterior_matrices(pair_matrices, 2),
        matrix_pair_n4_anticommutator(pair_matrices),
        atol=2e-12,
        rtol=2e-12,
    )


def test_diagonal_virtual_matrices_collapse_to_lc_agp() -> None:
    dimension = 7
    bond = 3
    diagonal_pairs = _random_complex((bond, dimension, dimension), 30)
    diagonal_pairs = diagonal_pairs - diagonal_pairs.transpose(1, 2)
    pair_matrices = torch.zeros(
        dimension, dimension, bond, bond, dtype=diagonal_pairs.dtype
    )
    indices = torch.arange(bond)
    pair_matrices[:, :, indices, indices] = diagonal_pairs.permute(1, 2, 0)
    left = _random_complex((bond,), 31)
    right = _random_complex((bond,), 32)

    observed = matrix_pair_exterior_coefficients(pair_matrices, 2, left, right)
    expected = sum(
        left[channel]
        * right[channel]
        * agp_exterior_coefficients(diagonal_pairs[channel], 2)
        for channel in range(bond)
    )
    torch.testing.assert_close(observed, expected, atol=2e-11, rtol=2e-11)


def test_virtual_similarity_gauge_leaves_state_invariant() -> None:
    pair_matrices = _random_skew_matrix_pairs(6, 2, 40)
    left = _random_complex((2,), 41)
    right = _random_complex((2,), 42)
    gauge = _random_complex((2, 2), 43) + 2 * torch.eye(2)
    inverse = torch.linalg.inv(gauge)
    transformed = torch.einsum(
        "ab,ijbc,cd->ijad", inverse, pair_matrices, gauge
    )
    torch.testing.assert_close(
        matrix_pair_exterior_coefficients(pair_matrices, 2, left, right),
        matrix_pair_exterior_coefficients(
            transformed, 2, left @ gauge, inverse @ right
        ),
        atol=3e-11,
        rtol=3e-11,
    )


def test_recurrence_is_autodifferentiable() -> None:
    pair_matrices = _random_skew_matrix_pairs(5, 2, 50).requires_grad_()
    left = _random_complex((2,), 51)
    right = _random_complex((2,), 52)
    coefficients = matrix_pair_exterior_coefficients(pair_matrices, 2, left, right)
    loss = torch.vdot(coefficients, coefficients).real
    gradient = torch.autograd.grad(loss, pair_matrices)[0]
    assert torch.isfinite(gradient).all()


def test_n4_norm_and_one_body_match_full_particle_tensor() -> None:
    pair_matrices = _random_skew_matrix_pairs(5, 2, 60)
    left = _random_complex((2,), 61)
    right = _random_complex((2,), 62)
    raw_operator = _random_complex((5, 5), 63)
    operator = raw_operator + raw_operator.conj().transpose(0, 1)
    tensor = matrix_pair_tensor(pair_matrices, 2, left, right)

    torch.testing.assert_close(
        matrix_pair_norm(pair_matrices, 2, left, right),
        torch.vdot(tensor.reshape(-1), tensor.reshape(-1)).real,
        atol=2e-10,
        rtol=2e-12,
    )
    torch.testing.assert_close(
        matrix_pair_one_body_expectation(
            pair_matrices, 2, left, right, operator
        ),
        torch.vdot(
            tensor.reshape(-1), apply_one_body_sum(tensor, operator).reshape(-1)
        ),
        atol=3e-9,
        rtol=3e-12,
    )

    coefficients = matrix_pair_exterior_coefficients(pair_matrices, 2, left, right)
    density = one_body_density_exterior_coefficients(coefficients, 5, 4)
    torch.testing.assert_close(
        torch.trace(operator @ density)
        * torch.vdot(coefficients, coefficients).real,
        matrix_pair_one_body_expectation(
            pair_matrices, 2, left, right, operator
        ),
        atol=3e-9,
        rtol=3e-12,
    )


def test_correlation_multiplicity_is_one_for_a_single_slater() -> None:
    dimension = 7
    particles = 4
    orbitals = _random_complex((dimension, particles), 70)
    orbitals = torch.linalg.qr(orbitals).Q
    supports = list(itertools.combinations(range(dimension), particles))
    coefficients = torch.stack(
        [torch.linalg.det(orbitals[list(support)]) for support in supports]
    )
    torch.testing.assert_close(
        fermionic_correlation_multiplicity(
            coefficients, dimension, particles
        ),
        torch.ones((), dtype=torch.float64),
        atol=3e-12,
        rtol=3e-12,
    )


@torch.no_grad()
def test_shift_tags_reduce_symmetrized_top_form_to_cayley_determinant() -> None:
    for order in (1, 2, 3):
        entries = _random_complex((order, order, 2, 2), 80 + order)
        left_block = _random_complex((2,), 90 + order)
        right_block = _random_complex((2,), 100 + order)
        pair_matrices, left, right = tagged_cayley_pair_data(
            entries, left_block, right_block
        )
        coefficient = matrix_pair_exterior_coefficients(
            pair_matrices, order, left, right
        )
        assert coefficient.numel() == 1
        torch.testing.assert_close(
            coefficient[0],
            tagged_cayley_expected_amplitude(entries, left_block, right_block),
            atol=2e-11,
            rtol=2e-11,
        )


def test_matrix_pair_power_has_polynomial_bond_matrix_wedge_embedding() -> None:
    pair_matrices = _random_skew_matrix_pairs(6, 2, 120)
    left = _random_complex((2,), 121)
    right = _random_complex((2,), 122)
    cores = matrix_pair_femps_cores(pair_matrices, 2, left, right)
    assert [core.shape[0] for core in cores] == [1, 12, 2, 12]
    assert [core.shape[2] for core in cores] == [12, 2, 12, 1]
    torch.testing.assert_close(
        femps_exterior_coefficients(cores),
        matrix_pair_exterior_coefficients(pair_matrices, 2, left, right),
        atol=3e-10,
        rtol=3e-11,
    )
