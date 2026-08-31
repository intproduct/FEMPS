import math

import torch

from femps.exterior import (
    exterior_dynamic_program_cost,
    femps_norm_exterior,
    femps_norm_paths,
    materialize_femps_matrix,
    one_body_expectation_explicit,
    one_body_expectation_paths,
    two_body_expectation_explicit,
    two_body_expectation_paths,
)


def _random_complex(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imag = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imag)


def _cores() -> list[torch.Tensor]:
    return [
        _random_complex((1, 4, 2), 51),
        _random_complex((2, 4, 2), 52),
        _random_complex((2, 4, 1), 53),
    ]


def _interaction() -> torch.Tensor:
    weights = torch.tensor(
        [
            [0.0, 0.3, -0.2, 0.5],
            [0.3, 0.0, 0.7, -0.1],
            [-0.2, 0.7, 0.0, 0.4],
            [0.5, -0.1, 0.4, 0.0],
        ],
        dtype=torch.float64,
    )
    interaction = torch.zeros(4, 4, 4, 4, dtype=torch.complex128)
    for first in range(4):
        for second in range(4):
            interaction[first, second, first, second] = weights[first, second]
    return interaction


def _gradients(function) -> tuple[torch.Tensor, ...]:
    variables = [core.detach().clone().requires_grad_(True) for core in _cores()]
    loss = function(variables).real
    return torch.autograd.grad(loss, variables)


def test_three_independent_norm_contractions_agree() -> None:
    cores = _cores()
    state = materialize_femps_matrix(cores)
    full_norm = torch.vdot(state.reshape(-1), state.reshape(-1)).real
    exterior_norm = femps_norm_exterior(cores)
    path_norm = femps_norm_paths(cores)

    torch.testing.assert_close(exterior_norm, full_norm, atol=2e-10, rtol=2e-13)
    torch.testing.assert_close(path_norm.real, full_norm, atol=2e-10, rtol=2e-13)
    assert path_norm.imag.abs().item() < 2e-10


def test_one_body_path_cofactors_match_full_tensor() -> None:
    cores = _cores()
    raw = _random_complex((4, 4), 61)
    operator = raw + raw.conj().transpose(0, 1)
    explicit = one_body_expectation_explicit(cores, operator)
    paths = one_body_expectation_paths(cores, operator)
    torch.testing.assert_close(paths, explicit, atol=2e-9, rtol=2e-13)


def test_two_body_second_cofactors_match_full_tensor() -> None:
    cores = _cores()
    interaction = _interaction()
    explicit = two_body_expectation_explicit(cores, interaction)
    paths = two_body_expectation_paths(cores, interaction)
    torch.testing.assert_close(paths, explicit, atol=3e-9, rtol=3e-13)


def test_norm_gradients_agree_across_all_three_routes() -> None:
    full = _gradients(
        lambda cores: torch.vdot(
            materialize_femps_matrix(cores).reshape(-1),
            materialize_femps_matrix(cores).reshape(-1),
        )
    )
    exterior = _gradients(femps_norm_exterior)
    paths = _gradients(femps_norm_paths)
    for full_gradient, exterior_gradient, path_gradient in zip(full, exterior, paths):
        torch.testing.assert_close(exterior_gradient, full_gradient, atol=2e-9, rtol=2e-12)
        torch.testing.assert_close(path_gradient, full_gradient, atol=2e-9, rtol=2e-12)


def test_operator_gradients_agree_between_explicit_and_path_routes() -> None:
    raw = _random_complex((4, 4), 61)
    operator = raw + raw.conj().transpose(0, 1)
    explicit_one = _gradients(lambda cores: one_body_expectation_explicit(cores, operator))
    path_one = _gradients(lambda cores: one_body_expectation_paths(cores, operator))
    explicit_two = _gradients(
        lambda cores: two_body_expectation_explicit(cores, _interaction())
    )
    path_two = _gradients(lambda cores: two_body_expectation_paths(cores, _interaction()))
    for explicit_gradient, path_gradient in zip(explicit_one, path_one):
        torch.testing.assert_close(path_gradient, explicit_gradient, atol=3e-8, rtol=3e-12)
    for explicit_gradient, path_gradient in zip(explicit_two, path_two):
        torch.testing.assert_close(path_gradient, explicit_gradient, atol=3e-8, rtol=3e-12)


def test_exterior_dynamic_program_cost_matches_closed_count() -> None:
    operations, peak = exterior_dynamic_program_cost(6, (1, 2, 3, 2, 1))
    expected = 2 * 6
    expected += 2 * 3 * 2 * math.comb(6, 2)
    expected += 3 * 2 * 3 * math.comb(6, 3)
    expected += 2 * 1 * 4 * math.comb(6, 4)
    assert operations == expected
    assert peak == max(2 * 6, 3 * math.comb(6, 2), 2 * math.comb(6, 3), math.comb(6, 4))
