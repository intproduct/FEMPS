"""Exact rational verifier for the T2 matrix-pair LC-AGP collapse.

This script imports neither PyTorch nor ``femps``.  It raises a symbolic
2 x 2 upper-triangular matrix of commuting variables to powers, constructs a
rational power-sum interpolation on the total-degree triangular grid, and
checks the two polynomials coefficient by coefficient.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import TypeAlias

Polynomial: TypeAlias = dict[tuple[int, int, int], Fraction]


def _add(first: Polynomial, second: Polynomial) -> Polynomial:
    result = dict(first)
    for exponent, value in second.items():
        updated = result.get(exponent, Fraction()) + value
        if updated:
            result[exponent] = updated
        else:
            result.pop(exponent, None)
    return result


def _scale(polynomial: Polynomial, scalar: Fraction) -> Polynomial:
    return {
        exponent: scalar * value
        for exponent, value in polynomial.items()
        if scalar * value
    }


def _multiply(first: Polynomial, second: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for first_exponent, first_value in first.items():
        for second_exponent, second_value in second.items():
            exponent = tuple(
                left + right
                for left, right in zip(
                    first_exponent, second_exponent, strict=True
                )
            )
            result[exponent] = (
                result.get(exponent, Fraction())
                + first_value * second_value
            )
    return {exponent: value for exponent, value in result.items() if value}


def _matrix_multiply(
    first: list[list[Polynomial]], second: list[list[Polynomial]]
) -> list[list[Polynomial]]:
    return [
        [
            _add(
                _multiply(first[row][0], second[0][column]),
                _multiply(first[row][1], second[1][column]),
            )
            for column in range(2)
        ]
        for row in range(2)
    ]


def _fraction_solve(
    matrix: list[list[Fraction]], right_hand_side: list[Fraction]
) -> list[Fraction]:
    order = len(matrix)
    augmented = [
        [*row, right_hand_side[index]] for index, row in enumerate(matrix)
    ]
    for column in range(order):
        pivot = next(
            row
            for row in range(column, order)
            if augmented[row][column]
        )
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        pivot_value = augmented[column][column]
        augmented[column] = [
            value / pivot_value for value in augmented[column]
        ]
        for row in range(order):
            if row == column or not augmented[row][column]:
                continue
            scale = augmented[row][column]
            augmented[row] = [
                value - scale * pivot_entry
                for value, pivot_entry in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[index][-1] for index in range(order)]


def _triangular_grid(order: int) -> list[tuple[int, int]]:
    return [
        (g_value, total - g_value)
        for total in range(order + 1)
        for g_value in range(total + 1)
    ]


def _power_polynomial(order: int, g_value: int, h_value: int) -> Polynomial:
    result: Polynomial = {}
    factorial = math.factorial
    for g_exponent, h_exponent in _triangular_grid(order):
        f_exponent = order - g_exponent - h_exponent
        coefficient = Fraction(
            factorial(order)
            * g_value**g_exponent
            * h_value**h_exponent,
            factorial(f_exponent)
            * factorial(g_exponent)
            * factorial(h_exponent),
        )
        if coefficient:
            result[(f_exponent, g_exponent, h_exponent)] = coefficient
    return result


def _target_polynomial(order: int) -> Polynomial:
    return {
        (order - 1 - h_exponent, 1, h_exponent): Fraction(1)
        for h_exponent in range(order)
    }


def _decomposition(order: int) -> tuple[list[tuple[int, int]], list[Fraction]]:
    grid = _triangular_grid(order)
    exponents = [
        (order - g_exponent - h_exponent, g_exponent, h_exponent)
        for g_exponent, h_exponent in grid
    ]
    powers = [_power_polynomial(order, *point) for point in grid]
    matrix = [
        [power.get(exponent, Fraction()) for power in powers]
        for exponent in exponents
    ]
    target = _target_polynomial(order)
    right_hand_side = [target.get(exponent, Fraction()) for exponent in exponents]
    return grid, _fraction_solve(matrix, right_hand_side)


def _matrix_power_offdiagonal(order: int) -> Polynomial:
    zero: Polynomial = {}
    one: Polynomial = {(0, 0, 0): Fraction(1)}
    f: Polynomial = {(1, 0, 0): Fraction(1)}
    g: Polynomial = {(0, 1, 0): Fraction(1)}
    h: Polynomial = {(0, 0, 1): Fraction(1)}
    value = [[one, zero], [zero, one]]
    generator = [[f, g], [zero, h]]
    for _ in range(order):
        value = _matrix_multiply(value, generator)
    if value[0][0] != {(order, 0, 0): Fraction(1)}:
        raise AssertionError("left diagonal matrix-power identity failed")
    if value[1][1] != {(0, 0, order): Fraction(1)}:
        raise AssertionError("right diagonal matrix-power identity failed")
    if value[1][0]:
        raise AssertionError("upper-triangular matrix power developed a lower block")
    return value[0][1]


def _fraction_payload(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _polynomial_payload(polynomial: Polynomial) -> list[list[object]]:
    return [
        [*exponent, _fraction_payload(value)]
        for exponent, value in sorted(polynomial.items())
    ]


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_certificate(max_order: int = 6) -> dict[str, object]:
    cases = []
    for order in range(1, max_order + 1):
        grid, weights = _decomposition(order)
        observed: Polynomial = {}
        for point, weight in zip(grid, weights, strict=True):
            observed = _add(
                observed,
                _scale(_power_polynomial(order, *point), weight),
            )
        target = _target_polynomial(order)
        matrix_power = _matrix_power_offdiagonal(order)
        if observed != target or matrix_power != target:
            raise AssertionError(f"triangular collapse failed at order {order}")
        nonzero_weights = [weight for weight in weights if weight]
        cases.append(
            {
                "pairs": order,
                "triangular_grid_terms": len(grid),
                "nonzero_lc_agp_terms_for_offdiagonal": len(nonzero_weights),
                "arbitrary_boundary_term_bound": len(grid) + 2,
                "maximum_weight_numerator_bits": max(
                    abs(weight.numerator).bit_length()
                    for weight in nonzero_weights
                ),
                "maximum_weight_denominator_bits": max(
                    weight.denominator.bit_length()
                    for weight in nonzero_weights
                ),
                "weights_sha256": _digest(
                    [_fraction_payload(weight) for weight in weights]
                ),
                "target_polynomial_sha256": _digest(
                    _polynomial_payload(target)
                ),
            }
        )
    body: dict[str, object] = {
        "schema": "femps.triangular-pair-lc-agp-collapse.v1",
        "arithmetic": "exact rational",
        "identity": (
            "([[F,G],[0,H]]^M)[0,1] = sum_(t=0)^(M-1) "
            "F^t G H^(M-1-t) = rational sum of (F+bG+cH)^M"
        ),
        "orders": list(range(1, max_order + 1)),
        "cases": cases,
    }
    body["certificate_sha256"] = _digest(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-order", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.verify is not None and arguments.output is not None:
        raise SystemExit("choose either --verify or --output")
    observed = build_certificate(arguments.max_order)
    if arguments.verify is not None:
        expected = json.loads(arguments.verify.read_text(encoding="utf-8"))
        if observed != expected:
            raise SystemExit("certificate mismatch")
        print(
            f"verified {arguments.verify} "
            f"({observed['certificate_sha256']})"
        )
        return
    payload = json.dumps(observed, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
