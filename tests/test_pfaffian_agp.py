import math

import torch

from femps.exterior import (
    agp_exterior_coefficients,
    agp_femps_cores,
    agp_log_norm,
    agp_norm_generating,
    agp_one_body_expectation,
    agp_overlap_generating,
    agp_structural_counts,
    agp_sum_norm,
    agp_sum_one_body_expectation,
    agp_sum_two_body_expectation_factorized,
    agp_tensor,
    agp_two_body_expectation_factorized,
    apply_one_body_sum,
    apply_two_body_sum,
    blocked_agp_exterior_coefficients,
    blocked_agp_femps_cores,
    blocked_agp_norm,
    blocked_agp_one_body_expectation,
    blocked_agp_overlap,
    blocked_agp_tensor,
    blocked_agp_two_body_expectation_factorized,
    materialize_femps_matrix,
    femps_exterior_coefficients,
    pair_matrix_from_channels,
    pfaffian_recursive,
    real_skew_pair_decomposition,
)


def _random_complex(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imag = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imag)


def _channels() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = _random_complex((3, 6), 71) / 3.0
    right = _random_complex((3, 6), 72) / 3.0
    weights = _random_complex((3,), 73) / 2.0
    return left, right, weights


def test_recursive_pfaffian_matches_closed_four_by_four_and_determinant() -> None:
    raw = _random_complex((4, 4), 70)
    skew = raw - raw.transpose(0, 1)
    expected = (
        skew[0, 1] * skew[2, 3]
        - skew[0, 2] * skew[1, 3]
        + skew[0, 3] * skew[1, 2]
    )
    observed = pfaffian_recursive(skew)
    torch.testing.assert_close(observed, expected, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(observed * observed, torch.linalg.det(skew), atol=2e-12, rtol=2e-12)


def test_ordered_channel_femps_equals_pfaffian_agp_tensor() -> None:
    left, right, weights = _channels()
    pair_matrix = pair_matrix_from_channels(left, right, weights)
    by_pfaffians = agp_tensor(pair_matrix, pairs=2)
    by_femps = materialize_femps_matrix(
        agp_femps_cores(left, right, pairs=2, weights=weights)
    )
    torch.testing.assert_close(by_femps, by_pfaffians, atol=3e-12, rtol=3e-12)


def test_ordered_channel_embedding_for_three_pairs_in_exterior_coordinates() -> None:
    left = _random_complex((4, 8), 80) / 4.0
    right = _random_complex((4, 8), 81) / 4.0
    weights = _random_complex((4,), 82) / 2.0
    pair_matrix = pair_matrix_from_channels(left, right, weights)
    by_pfaffians = agp_exterior_coefficients(pair_matrix, pairs=3)
    by_femps = femps_exterior_coefficients(
        agp_femps_cores(left, right, pairs=3, weights=weights)
    )
    torch.testing.assert_close(by_femps, by_pfaffians, atol=3e-11, rtol=3e-11)


def test_generating_overlap_matches_pfaffian_coefficient_inner_product() -> None:
    left, right, weights = _channels()
    bra = pair_matrix_from_channels(left, right, weights)
    ket_left = _random_complex((3, 6), 74) / 3.0
    ket_right = _random_complex((3, 6), 75) / 3.0
    ket_weights = _random_complex((3,), 76) / 2.0
    ket = pair_matrix_from_channels(ket_left, ket_right, ket_weights)
    bra_coefficients = agp_exterior_coefficients(bra, pairs=2)
    ket_coefficients = agp_exterior_coefficients(ket, pairs=2)
    explicit = torch.vdot(bra_coefficients, ket_coefficients)
    generating = agp_overlap_generating(bra, ket, pairs=2)
    torch.testing.assert_close(generating, explicit, atol=3e-11, rtol=3e-12)


def test_polynomial_norm_matches_pfaffians_and_full_tensor() -> None:
    left, right, weights = _channels()
    pair_matrix = pair_matrix_from_channels(left, right, weights)
    coefficients = agp_exterior_coefficients(pair_matrix, pairs=2)
    state = agp_tensor(pair_matrix, pairs=2)
    expected = torch.vdot(coefficients, coefficients).real
    torch.testing.assert_close(agp_norm_generating(pair_matrix, 2), expected, atol=3e-11, rtol=3e-12)
    torch.testing.assert_close(torch.vdot(state.reshape(-1), state.reshape(-1)).real, expected)


def test_polynomial_one_body_expectation_matches_full_tensor() -> None:
    left, right, weights = _channels()
    pair_matrix = pair_matrix_from_channels(left, right, weights)
    state = agp_tensor(pair_matrix, pairs=2)
    raw = _random_complex((6, 6), 77)
    operator = raw + raw.conj().transpose(0, 1)
    explicit = torch.vdot(
        state.reshape(-1), apply_one_body_sum(state, operator).reshape(-1)
    )
    polynomial = agp_one_body_expectation(pair_matrix, 2, operator)
    torch.testing.assert_close(polynomial, explicit, atol=4e-10, rtol=4e-12)


def test_factorized_two_body_expectation_matches_full_tensor() -> None:
    left, right, weights = _channels()
    pair_matrix = pair_matrix_from_channels(left, right, weights)
    state = agp_tensor(pair_matrix, pairs=2)
    raw_left = _random_complex((2, 6, 6), 78)
    raw_right = _random_complex((2, 6, 6), 79)
    left_operators = raw_left + raw_left.conj().transpose(1, 2)
    right_operators = raw_right + raw_right.conj().transpose(1, 2)
    term_weights = torch.tensor([0.3, -0.2], dtype=torch.complex128)
    interaction = torch.zeros(6, 6, 6, 6, dtype=torch.complex128)
    for term in range(2):
        direct = torch.einsum(
            "pr,qs->pqrs", left_operators[term], right_operators[term]
        )
        swapped = torch.einsum(
            "pr,qs->pqrs", right_operators[term], left_operators[term]
        )
        interaction = interaction + 0.5 * term_weights[term] * (direct + swapped)
    explicit = torch.vdot(
        state.reshape(-1), apply_two_body_sum(state, interaction).reshape(-1)
    )
    polynomial = agp_two_body_expectation_factorized(
        pair_matrix,
        2,
        left_operators,
        right_operators,
        term_weights,
    )
    torch.testing.assert_close(polynomial, explicit, atol=2e-8, rtol=2e-11)


def test_finite_agp_sum_norm_and_operators_match_explicit_state() -> None:
    left, right, weights = _channels()
    first = pair_matrix_from_channels(left, right, weights)
    second = pair_matrix_from_channels(
        _random_complex((3, 6), 83) / 3.0,
        _random_complex((3, 6), 84) / 3.0,
        _random_complex((3,), 85) / 2.0,
    )
    matrices = torch.stack((first, second))
    amplitudes = torch.tensor([0.7 + 0.2j, -0.3 + 0.1j], dtype=torch.complex128)
    state = amplitudes[0] * agp_tensor(first, 2) + amplitudes[1] * agp_tensor(second, 2)

    explicit_norm = torch.vdot(state.reshape(-1), state.reshape(-1)).real
    torch.testing.assert_close(
        agp_sum_norm(matrices, amplitudes, 2),
        explicit_norm,
        atol=3e-10,
        rtol=3e-11,
    )

    raw_operator = _random_complex((6, 6), 86)
    operator = raw_operator + raw_operator.conj().transpose(0, 1)
    explicit_one = torch.vdot(
        state.reshape(-1), apply_one_body_sum(state, operator).reshape(-1)
    )
    torch.testing.assert_close(
        agp_sum_one_body_expectation(matrices, amplitudes, 2, operator),
        explicit_one,
        atol=3e-9,
        rtol=3e-10,
    )

    raw_left = _random_complex((1, 6, 6), 87)
    raw_right = _random_complex((1, 6, 6), 88)
    left_factors = raw_left + raw_left.conj().transpose(1, 2)
    right_factors = raw_right + raw_right.conj().transpose(1, 2)
    factor_weights = torch.tensor([0.15], dtype=torch.complex128)
    direct = torch.einsum("pr,qs->pqrs", left_factors[0], right_factors[0])
    swapped = torch.einsum("pr,qs->pqrs", right_factors[0], left_factors[0])
    interaction = 0.5 * factor_weights[0] * (direct + swapped)
    explicit_two = torch.vdot(
        state.reshape(-1), apply_two_body_sum(state, interaction).reshape(-1)
    )
    torch.testing.assert_close(
        agp_sum_two_body_expectation_factorized(
            matrices,
            amplitudes,
            2,
            left_factors,
            right_factors,
            factor_weights,
        ),
        explicit_two,
        atol=3e-8,
        rtol=3e-9,
    )


def _channel_gradients(use_polynomial: bool) -> tuple[torch.Tensor, ...]:
    left, right, weights = [value.detach().clone().requires_grad_(True) for value in _channels()]
    if use_polynomial:
        pair_matrix = pair_matrix_from_channels(left, right, weights)
        loss = agp_norm_generating(pair_matrix, pairs=2)
    else:
        state = materialize_femps_matrix(
            agp_femps_cores(left, right, pairs=2, weights=weights)
        )
        loss = torch.vdot(state.reshape(-1), state.reshape(-1)).real
    return torch.autograd.grad(loss, (left, right, weights))


def test_polynomial_norm_ad_gradients_match_femps_materialization() -> None:
    polynomial = _channel_gradients(True)
    explicit = _channel_gradients(False)
    for polynomial_gradient, explicit_gradient in zip(polynomial, explicit):
        torch.testing.assert_close(
            polynomial_gradient, explicit_gradient, atol=2e-9, rtol=2e-11
        )


def _pair_operator_gradient(kind: str, polynomial: bool) -> torch.Tensor:
    raw = _random_complex((6, 6), 89).requires_grad_(True)
    pair_matrix = (raw - raw.transpose(0, 1)) / 3.0
    operator_raw = _random_complex((6, 6), 90)
    operator = operator_raw + operator_raw.conj().transpose(0, 1)
    factor_left_raw = _random_complex((1, 6, 6), 91)
    factor_right_raw = _random_complex((1, 6, 6), 92)
    factor_left = factor_left_raw + factor_left_raw.conj().transpose(1, 2)
    factor_right = factor_right_raw + factor_right_raw.conj().transpose(1, 2)
    factor_weights = torch.tensor([0.12], dtype=torch.complex128)
    if kind == "one":
        if polynomial:
            value = agp_one_body_expectation(pair_matrix, 2, operator)
        else:
            state = agp_tensor(pair_matrix, 2)
            value = torch.vdot(
                state.reshape(-1), apply_one_body_sum(state, operator).reshape(-1)
            )
    else:
        if polynomial:
            value = agp_two_body_expectation_factorized(
                pair_matrix,
                2,
                factor_left,
                factor_right,
                factor_weights,
            )
        else:
            direct = torch.einsum(
                "pr,qs->pqrs", factor_left[0], factor_right[0]
            )
            swapped = torch.einsum(
                "pr,qs->pqrs", factor_right[0], factor_left[0]
            )
            interaction = 0.5 * factor_weights[0] * (direct + swapped)
            state = agp_tensor(pair_matrix, 2)
            value = torch.vdot(
                state.reshape(-1),
                apply_two_body_sum(state, interaction).reshape(-1),
            )
    return torch.autograd.grad(value.real, raw)[0]


def test_polynomial_operator_ad_gradients_match_explicit_tensor() -> None:
    for kind in ("one", "two"):
        polynomial = _pair_operator_gradient(kind, True)
        explicit = _pair_operator_gradient(kind, False)
        torch.testing.assert_close(polynomial, explicit, atol=3e-7, rtol=3e-9)


def test_agp_structural_counts_expose_exponential_slater_expansion() -> None:
    counts = agp_structural_counts(dimension=16, pairs=4, channels=8)
    assert counts["particles"] == 8
    assert counts["femps_internal_bond"] == 8
    assert counts["nonzero_ordered_paths"] == math.comb(8, 4)
    assert counts["unrestricted_path_bound"] == 8**7
    assert counts["exterior_coefficients"] == math.comb(16, 8)


def test_real_skew_pair_decomposition_reconstructs_by_channel_rank() -> None:
    raw = torch.randn(
        8, 8, generator=torch.Generator().manual_seed(93), dtype=torch.float64
    )
    skew = raw - raw.transpose(0, 1)
    left, right, weights = real_skew_pair_decomposition(skew, channels=4)
    reconstructed = pair_matrix_from_channels(left, right, weights)
    torch.testing.assert_close(reconstructed, skew, atol=2e-12, rtol=2e-12)


def test_stable_norm_and_log_norm_survive_large_top_sector() -> None:
    dimension = 64
    pairs = dimension // 2
    raw = torch.randn(
        dimension,
        dimension,
        generator=torch.Generator().manual_seed(94),
        dtype=torch.float64,
    )
    skew = raw - raw.transpose(0, 1)
    norm = agp_norm_generating(skew, pairs)
    log_norm = agp_log_norm(skew, pairs)
    _, reference_log_norm = torch.linalg.slogdet(skew)
    torch.testing.assert_close(
        norm,
        torch.exp(reference_log_norm),
        atol=0,
        rtol=8e-14,
    )
    torch.testing.assert_close(log_norm, reference_log_norm, atol=3e-13, rtol=0)

    for scale in (1e-12, 1e12):
        observed = agp_log_norm(scale * skew, pairs)
        expected = reference_log_norm + 2 * pairs * math.log(scale)
        torch.testing.assert_close(observed, expected, atol=4e-13, rtol=0)


def test_scaled_transition_overlap_handles_reciprocal_pair_scales() -> None:
    dimension = 12
    generator = torch.Generator().manual_seed(95)
    bra_raw = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    ket_raw = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    bra = bra_raw - bra_raw.transpose(0, 1)
    ket = ket_raw - ket_raw.transpose(0, 1)
    reference = agp_overlap_generating(bra, ket, pairs=4)
    observed = agp_overlap_generating(1e-120 * bra, 1e120 * ket, pairs=4)
    torch.testing.assert_close(observed, reference, atol=2e-10, rtol=2e-12)


def test_blocked_agp_coefficients_equal_ordered_femps_and_overlap() -> None:
    left = _random_complex((2, 5), 96) / 3
    right = _random_complex((2, 5), 97) / 3
    weights = _random_complex((2,), 98) / 2
    blocked = _random_complex((5,), 99) / 2
    pair_matrix = pair_matrix_from_channels(left, right, weights)
    explicit = blocked_agp_exterior_coefficients(pair_matrix, blocked, pairs=1)
    by_femps = femps_exterior_coefficients(
        blocked_agp_femps_cores(blocked, left, right, pairs=1, weights=weights)
    )
    torch.testing.assert_close(by_femps, explicit, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(
        blocked_agp_norm(pair_matrix, blocked, pairs=1),
        torch.vdot(explicit, explicit).real,
        atol=3e-11,
        rtol=3e-11,
    )

    ket_left = _random_complex((2, 5), 100) / 3
    ket_right = _random_complex((2, 5), 101) / 3
    ket_weights = _random_complex((2,), 102) / 2
    ket_blocked = _random_complex((5,), 103) / 2
    ket_pair = pair_matrix_from_channels(ket_left, ket_right, ket_weights)
    ket_coefficients = blocked_agp_exterior_coefficients(
        ket_pair, ket_blocked, pairs=1
    )
    torch.testing.assert_close(
        blocked_agp_overlap(
            pair_matrix, blocked, ket_pair, ket_blocked, pairs=1
        ),
        torch.vdot(explicit, ket_coefficients),
        atol=5e-11,
        rtol=5e-11,
    )


def test_blocked_agp_operators_match_explicit_odd_particle_tensor() -> None:
    left = _random_complex((2, 5), 104) / 3
    right = _random_complex((2, 5), 105) / 3
    pair_matrix = pair_matrix_from_channels(left, right)
    blocked = _random_complex((5,), 106) / 2
    state = blocked_agp_tensor(pair_matrix, blocked, pairs=1)

    raw_operator = _random_complex((5, 5), 107)
    operator = raw_operator + raw_operator.conj().transpose(0, 1)
    explicit_one = torch.vdot(
        state.reshape(-1), apply_one_body_sum(state, operator).reshape(-1)
    )
    polynomial_one = blocked_agp_one_body_expectation(
        pair_matrix, blocked, 1, operator
    )
    torch.testing.assert_close(polynomial_one, explicit_one, atol=3e-9, rtol=3e-10)

    raw_left = _random_complex((1, 5, 5), 108)
    raw_right = _random_complex((1, 5, 5), 109)
    left_factor = raw_left + raw_left.conj().transpose(1, 2)
    right_factor = raw_right + raw_right.conj().transpose(1, 2)
    factor_weights = torch.tensor([0.17], dtype=torch.complex128)
    direct = torch.einsum("pr,qs->pqrs", left_factor[0], right_factor[0])
    swapped = torch.einsum("pr,qs->pqrs", right_factor[0], left_factor[0])
    interaction = 0.5 * factor_weights[0] * (direct + swapped)
    explicit_two = torch.vdot(
        state.reshape(-1), apply_two_body_sum(state, interaction).reshape(-1)
    )
    polynomial_two = blocked_agp_two_body_expectation_factorized(
        pair_matrix,
        blocked,
        1,
        left_factor,
        right_factor,
        factor_weights,
    )
    torch.testing.assert_close(polynomial_two, explicit_two, atol=2e-8, rtol=2e-9)


def test_one_particle_blocked_state_is_independent_of_pair_matrix() -> None:
    raw = _random_complex((5, 5), 110)
    pair_matrix = raw - raw.transpose(0, 1)
    blocked = _random_complex((5,), 111)
    operator_raw = _random_complex((5, 5), 112)
    operator = operator_raw + operator_raw.conj().transpose(0, 1)
    torch.testing.assert_close(
        blocked_agp_norm(pair_matrix, blocked, pairs=0),
        torch.vdot(blocked, blocked).real,
        atol=3e-11,
        rtol=3e-11,
    )
    torch.testing.assert_close(
        blocked_agp_one_body_expectation(pair_matrix, blocked, 0, operator),
        torch.vdot(blocked, operator @ blocked),
        atol=3e-10,
        rtol=3e-10,
    )


def test_blocked_agp_norm_gradients_match_explicit_coefficients() -> None:
    raw_reference = _random_complex((5, 5), 113) / 3
    blocked_reference = _random_complex((5,), 114) / 2

    def gradients(polynomial: bool) -> tuple[torch.Tensor, torch.Tensor]:
        raw = raw_reference.detach().clone().requires_grad_(True)
        blocked = blocked_reference.detach().clone().requires_grad_(True)
        pair_matrix = raw - raw.transpose(0, 1)
        if polynomial:
            value = blocked_agp_norm(pair_matrix, blocked, pairs=1)
        else:
            coefficients = blocked_agp_exterior_coefficients(
                pair_matrix, blocked, pairs=1
            )
            value = torch.vdot(coefficients, coefficients).real
        return torch.autograd.grad(value, (raw, blocked))

    polynomial_gradients = gradients(True)
    explicit_gradients = gradients(False)
    for polynomial, explicit in zip(polynomial_gradients, explicit_gradients):
        torch.testing.assert_close(polynomial, explicit, atol=2e-9, rtol=2e-10)
