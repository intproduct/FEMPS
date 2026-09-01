"""Exact rational verifier for fixed Mat2 matrix-pair LC-AGP collapse.

The verifier treats the four entries of a symbolic 2 x 2 matrix as commuting
two-form variables.  It raises the matrix exactly, contracts deterministic
boundaries, and reconstructs the resulting homogeneous polynomial as a
rational sum of M-th powers on a three-dimensional simplex grid.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import TypeAlias

Exponent = tuple[int, int, int, int]
Polynomial: TypeAlias = dict[Exponent, Fraction]


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


def _simplex(order: int) -> list[tuple[int, int, int]]:
    return [
        (second, third, total - second - third)
        for total in range(order + 1)
        for second in range(total + 1)
        for third in range(total - second + 1)
    ]


def _power_polynomial(
    order: int, point: tuple[int, int, int]
) -> Polynomial:
    result: Polynomial = {}
    factorial = math.factorial
    for second, third, fourth in _simplex(order):
        first = order - second - third - fourth
        coefficient = Fraction(
            factorial(order)
            * point[0] ** second
            * point[1] ** third
            * point[2] ** fourth,
            factorial(first)
            * factorial(second)
            * factorial(third)
            * factorial(fourth),
        )
        if coefficient:
            result[(first, second, third, fourth)] = coefficient
    return result


def _contracted_matrix_power(order: int) -> Polynomial:
    zero: Polynomial = {}
    one: Polynomial = {(0, 0, 0, 0): Fraction(1)}
    x0: Polynomial = {(1, 0, 0, 0): Fraction(1)}
    x1: Polynomial = {(0, 1, 0, 0): Fraction(1)}
    x2: Polynomial = {(0, 0, 1, 0): Fraction(1)}
    x3: Polynomial = {(0, 0, 0, 1): Fraction(1)}
    value = [[one, zero], [zero, one]]
    generator = [[x0, x1], [x2, x3]]
    for _ in range(order):
        value = _matrix_multiply(value, generator)
    left = (Fraction(2), Fraction(3))
    right = (Fraction(5), Fraction(7))
    result: Polynomial = {}
    for row in range(2):
        for column in range(2):
            result = _add(
                result,
                _scale(value[row][column], left[row] * right[column]),
            )
    return result


def _decomposition(
    order: int, target: Polynomial
) -> tuple[list[tuple[int, int, int]], list[Fraction]]:
    grid = _simplex(order)
    exponents = [
        (order - second - third - fourth, second, third, fourth)
        for second, third, fourth in grid
    ]
    powers = [_power_polynomial(order, point) for point in grid]
    matrix = [
        [power.get(exponent, Fraction()) for power in powers]
        for exponent in exponents
    ]
    right_hand_side = [target.get(exponent, Fraction()) for exponent in exponents]
    return grid, _fraction_solve(matrix, right_hand_side)


def _fraction_payload(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_certificate(max_order: int = 4) -> dict[str, object]:
    cases = []
    for order in range(1, max_order + 1):
        target = _contracted_matrix_power(order)
        grid, weights = _decomposition(order, target)
        observed: Polynomial = {}
        for point, weight in zip(grid, weights, strict=True):
            observed = _add(
                observed,
                _scale(_power_polynomial(order, point), weight),
            )
        if observed != target:
            raise AssertionError(f"Mat2 collapse failed at order {order}")
        nonzero = [weight for weight in weights if weight]
        cases.append(
            {
                "pairs": order,
                "lc_agp_term_bound": math.comb(order + 3, 3),
                "nonzero_terms_for_deterministic_boundaries": len(nonzero),
                "maximum_weight_numerator_bits": max(
                    abs(weight.numerator).bit_length() for weight in nonzero
                ),
                "maximum_weight_denominator_bits": max(
                    weight.denominator.bit_length() for weight in nonzero
                ),
                "weights_sha256": _digest(
                    [_fraction_payload(weight) for weight in weights]
                ),
                "target_sha256": _digest(
                    [
                        [*exponent, _fraction_payload(value)]
                        for exponent, value in sorted(target.items())
                    ]
                ),
            }
        )
    body: dict[str, object] = {
        "schema": "femps.mat2-pair-lc-agp-collapse.v1",
        "arithmetic": "exact rational",
        "matrix": "[[x0,x1],[x2,x3]]",
        "boundaries": {"left": [2, 3], "right": [5, 7]},
        "term_bound": "binom(M+3,3)",
        "orders": list(range(1, max_order + 1)),
        "cases": cases,
    }
    body["certificate_sha256"] = _digest(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-order", type=int, default=4)
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
