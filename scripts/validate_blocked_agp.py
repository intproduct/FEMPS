"""Validate odd-particle blocked Pfaffian contractions and Blackwell parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from femps.devices import resolve_device
from femps.exterior import (
    apply_one_body_sum,
    apply_two_body_sum,
    blocked_agp_exterior_coefficients,
    blocked_agp_femps_cores,
    blocked_agp_norm,
    blocked_agp_one_body_expectation,
    blocked_agp_overlap,
    blocked_agp_tensor,
    blocked_agp_two_body_expectation_factorized,
    femps_exterior_coefficients,
    pair_matrix_from_channels,
)


def _random_complex(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imaginary)


def _small_truth_errors() -> dict[str, float]:
    dimension = 5
    pairs = 1
    left = _random_complex((2, dimension), 501) / 3
    right = _random_complex((2, dimension), 502) / 3
    weights = _random_complex((2,), 503) / 2
    blocked = _random_complex((dimension,), 504) / 2
    pair_matrix = pair_matrix_from_channels(left, right, weights)
    coefficients = blocked_agp_exterior_coefficients(pair_matrix, blocked, pairs)
    femps_coefficients = femps_exterior_coefficients(
        blocked_agp_femps_cores(blocked, left, right, pairs, weights)
    )
    explicit_norm = torch.vdot(coefficients, coefficients).real
    polynomial_norm = blocked_agp_norm(pair_matrix, blocked, pairs)

    ket_raw = _random_complex((dimension, dimension), 505) / 3
    ket_pair = ket_raw - ket_raw.transpose(0, 1)
    ket_blocked = _random_complex((dimension,), 506) / 2
    ket_coefficients = blocked_agp_exterior_coefficients(
        ket_pair, ket_blocked, pairs
    )
    explicit_overlap = torch.vdot(coefficients, ket_coefficients)
    polynomial_overlap = blocked_agp_overlap(
        pair_matrix, blocked, ket_pair, ket_blocked, pairs
    )

    state = blocked_agp_tensor(pair_matrix, blocked, pairs)
    operator_raw = _random_complex((dimension, dimension), 507)
    operator = operator_raw + operator_raw.conj().transpose(0, 1)
    explicit_one = torch.vdot(
        state.reshape(-1), apply_one_body_sum(state, operator).reshape(-1)
    )
    polynomial_one = blocked_agp_one_body_expectation(
        pair_matrix, blocked, pairs, operator
    )

    left_raw = _random_complex((1, dimension, dimension), 508)
    right_raw = _random_complex((1, dimension, dimension), 509)
    left_factor = left_raw + left_raw.conj().transpose(1, 2)
    right_factor = right_raw + right_raw.conj().transpose(1, 2)
    factor_weights = torch.tensor([0.11], dtype=torch.complex128)
    direct = torch.einsum("pr,qs->pqrs", left_factor[0], right_factor[0])
    swapped = torch.einsum("pr,qs->pqrs", right_factor[0], left_factor[0])
    interaction = 0.5 * factor_weights[0] * (direct + swapped)
    explicit_two = torch.vdot(
        state.reshape(-1), apply_two_body_sum(state, interaction).reshape(-1)
    )
    polynomial_two = blocked_agp_two_body_expectation_factorized(
        pair_matrix,
        blocked,
        pairs,
        left_factor,
        right_factor,
        factor_weights,
    )

    raw_reference = _random_complex((dimension, dimension), 510) / 3
    block_reference = _random_complex((dimension,), 511) / 2

    def norm_gradients(polynomial: bool) -> tuple[torch.Tensor, torch.Tensor]:
        raw = raw_reference.detach().clone().requires_grad_(True)
        block = block_reference.detach().clone().requires_grad_(True)
        matrix = raw - raw.transpose(0, 1)
        if polynomial:
            value = blocked_agp_norm(matrix, block, pairs)
        else:
            value_coefficients = blocked_agp_exterior_coefficients(
                matrix, block, pairs
            )
            value = torch.vdot(value_coefficients, value_coefficients).real
        return torch.autograd.grad(value, (raw, block))

    polynomial_gradients = norm_gradients(True)
    explicit_gradients = norm_gradients(False)
    gradient_error = max(
        float(torch.max(torch.abs(observed - expected)))
        for observed, expected in zip(polynomial_gradients, explicit_gradients)
    )
    return {
        "femps_coefficient_max_abs_error": float(
            torch.max(torch.abs(femps_coefficients - coefficients)).detach()
        ),
        "norm_absolute_error": float(
            torch.abs(polynomial_norm - explicit_norm).detach()
        ),
        "overlap_absolute_error": float(
            torch.abs(polynomial_overlap - explicit_overlap).detach()
        ),
        "one_body_absolute_error": float(
            torch.abs(polynomial_one - explicit_one).detach()
        ),
        "two_body_absolute_error": float(
            torch.abs(polynomial_two - explicit_two).detach()
        ),
        "norm_gradient_max_abs_error": gradient_error,
    }


def _energy_and_gradients(
    raw_pair: torch.Tensor,
    blocked: torch.Tensor,
    operator: torch.Tensor,
    pairs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dimension = raw_pair.shape[0]
    pair_matrix = (raw_pair - raw_pair.transpose(0, 1)) / dimension**0.5
    blocked_orbital = blocked / dimension**0.5
    norm = blocked_agp_norm(pair_matrix, blocked_orbital, pairs)
    numerator = blocked_agp_one_body_expectation(
        pair_matrix, blocked_orbital, pairs, operator
    )
    energy = (numerator / norm).real
    pair_gradient, block_gradient = torch.autograd.grad(
        energy, (raw_pair, blocked)
    )
    return energy, pair_gradient, block_gradient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/blocked_agp_validation.json"),
    )
    args = parser.parse_args()
    if 2 * args.pairs + 1 > args.dimension:
        raise ValueError("require 2*pairs+1 <= dimension")
    gpu = resolve_device("auto")
    if gpu.type != "cuda":
        raise RuntimeError("no supported CUDA device is available")

    raw_cpu = _random_complex((args.dimension, args.dimension), 512).requires_grad_(True)
    blocked_cpu = _random_complex((args.dimension,), 513).requires_grad_(True)
    operator_raw = _random_complex((args.dimension, args.dimension), 514)
    operator_cpu = (
        operator_raw + operator_raw.conj().transpose(0, 1)
    ) / args.dimension**0.5
    raw_gpu = raw_cpu.detach().to(gpu).requires_grad_(True)
    blocked_gpu = blocked_cpu.detach().to(gpu).requires_grad_(True)
    operator_gpu = operator_cpu.to(gpu)

    cpu_energy, cpu_pair_gradient, cpu_block_gradient = _energy_and_gradients(
        raw_cpu, blocked_cpu, operator_cpu, args.pairs
    )
    gpu_energy, gpu_pair_gradient, gpu_block_gradient = _energy_and_gradients(
        raw_gpu, blocked_gpu, operator_gpu, args.pairs
    )
    torch.cuda.synchronize(gpu)
    record = {
        "schema_version": 1,
        "experiment": "blocked_agp_exact_and_gpu_validation",
        "small_truth": {"D": 5, "N": 3, **_small_truth_errors()},
        "gpu_parity": {
            "D": args.dimension,
            "N": 2 * args.pairs + 1,
            "pairs": args.pairs,
            "dtype": "complex128",
            "device": str(gpu),
            "gpu": torch.cuda.get_device_name(gpu),
            "compute_capability": torch.cuda.get_device_capability(gpu),
            "cpu_energy": float(cpu_energy.detach()),
            "gpu_energy": float(gpu_energy.detach().cpu()),
            "energy_absolute_difference": float(
                torch.abs(cpu_energy.detach() - gpu_energy.detach().cpu())
            ),
            "pair_gradient_max_abs_difference": float(
                torch.max(
                    torch.abs(cpu_pair_gradient - gpu_pair_gradient.detach().cpu())
                )
            ),
            "blocked_gradient_max_abs_difference": float(
                torch.max(
                    torch.abs(cpu_block_gradient - gpu_block_gradient.detach().cpu())
                )
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    parity = record["gpu_parity"]
    passed = (
        max(value for key, value in record["small_truth"].items() if key.endswith("error"))
        < 1e-7
        and parity["energy_absolute_difference"] < 1e-9
        and parity["pair_gradient_max_abs_difference"] < 1e-8
        and parity["blocked_gradient_max_abs_difference"] < 1e-8
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
