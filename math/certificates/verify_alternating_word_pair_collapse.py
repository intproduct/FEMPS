"""Exact verifier for the alternating-word algebra LC-AGP collapse.

The algebra has generators x,y, relations x^2=y^2=0, and all words of length
d equal to zero.  The verifier compares direct algebra powers with the nested
exact decomposition induced by its embedding in Mat2(Q[z]/(z^d)).
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator, TypeAlias

Word = tuple[int, ...]
Exponent = tuple[int, ...]
Polynomial: TypeAlias = dict[Exponent, Fraction]


def _words(depth: int) -> list[Word]:
    result: list[Word] = [()]
    for length in range(1, depth):
        result.append(tuple(index % 2 for index in range(length)))
        result.append(tuple(1 - index % 2 for index in range(length)))
    return result


def _word_payload(word: Word) -> str:
    if not word:
        return "1"
    return "".join("x" if letter == 0 else "y" for letter in word)


def _word_product(first: Word, second: Word, depth: int) -> Word | None:
    if not first:
        return second
    if not second:
        return first
    if len(first) + len(second) >= depth or first[-1] == second[0]:
        return None
    return (*first, *second)


def _exponents(total: int, variables: int) -> Iterator[Exponent]:
    if variables == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _exponents(total - first, variables - 1):
            yield (first, *rest)


def _multinomial(order: int, exponent: Exponent) -> int:
    result = math.factorial(order)
    for power in exponent:
        result //= math.factorial(power)
    return result


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


def _direct_power(order: int, depth: int) -> dict[Word, Polynomial]:
    words = _words(depth)
    variables = len(words)
    zero_exponent = (0,) * variables
    value: dict[Word, Polynomial] = {(): {zero_exponent: Fraction(1)}}
    generator = {
        word: {
            tuple(int(index == variable) for index in range(variables)): Fraction(1)
        }
        for variable, word in enumerate(words)
    }
    for _ in range(order):
        updated: dict[Word, Polynomial] = {}
        for first_word, first_polynomial in value.items():
            for second_word, second_polynomial in generator.items():
                product = _word_product(first_word, second_word, depth)
                if product is None:
                    continue
                updated[product] = _add(
                    updated.get(product, {}),
                    _multiply(first_polynomial, second_polynomial),
                )
        value = updated
    return value


def _embedding_coordinate(word: Word) -> tuple[int, int, int]:
    if not word:
        raise ValueError("the identity embeds in both diagonal entries")
    return word[0], 1 - word[-1], len(word)


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


def _coefficient_weights(order: int, depth: int, degree: int) -> list[Fraction]:
    term_bound = order * (depth - 1) + 1
    nodes = range(term_bound)
    matrix = [
        [Fraction(node**power) for node in nodes]
        for power in range(term_bound)
    ]
    right_hand_side = [
        Fraction(power == degree) for power in range(term_bound)
    ]
    return _fraction_solve(matrix, right_hand_side)


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


def _mat2_entry_power(order: int, row: int, column: int) -> Polynomial:
    zero: Polynomial = {}
    one: Polynomial = {(0, 0, 0, 0): Fraction(1)}
    x = [
        {tuple(int(index == variable) for index in range(4)): Fraction(1)}
        for variable in range(4)
    ]
    value = [[one, zero], [zero, one]]
    generator = [[x[0], x[1]], [x[2], x[3]]]
    for _ in range(order):
        value = _matrix_multiply(value, generator)
    return value[row][column]


def _simplex(order: int) -> list[tuple[int, int, int]]:
    return [
        (second, third, total - second - third)
        for total in range(order + 1)
        for second in range(total + 1)
        for third in range(total - second + 1)
    ]


def _mat2_power_polynomial(
    order: int, point: tuple[int, int, int]
) -> Polynomial:
    result: Polynomial = {}
    for second, third, fourth in _simplex(order):
        first = order - second - third - fourth
        coefficient = Fraction(
            _multinomial(order, (first, second, third, fourth))
            * point[0] ** second
            * point[1] ** third
            * point[2] ** fourth
        )
        if coefficient:
            result[(first, second, third, fourth)] = coefficient
    return result


def _mat2_weights(order: int, row: int, column: int) -> list[Fraction]:
    grid = _simplex(order)
    exponents = [
        (order - second - third - fourth, second, third, fourth)
        for second, third, fourth in grid
    ]
    powers = [_mat2_power_polynomial(order, point) for point in grid]
    matrix = [
        [power.get(exponent, Fraction()) for power in powers]
        for exponent in exponents
    ]
    target = _mat2_entry_power(order, row, column)
    right_hand_side = [target.get(exponent, Fraction()) for exponent in exponents]
    return _fraction_solve(matrix, right_hand_side)


def _embedded_linear_form(
    depth: int, node: int, point: tuple[int, int, int]
) -> list[Fraction]:
    coefficients = [Fraction(0)] * (2 * depth - 1)
    coefficients[0] = Fraction(1 + point[2])
    for index, word in enumerate(_words(depth)[1:], start=1):
        row, column, degree = _embedding_coordinate(word)
        matrix_coordinate = 2 * row + column
        multiplier = (Fraction(1), *map(Fraction, point))[matrix_coordinate]
        coefficients[index] = multiplier * node**degree
    return coefficients


def _linear_form_power(order: int, coefficients: list[Fraction]) -> Polynomial:
    result: Polynomial = {}
    for exponent in _exponents(order, len(coefficients)):
        coefficient = Fraction(_multinomial(order, exponent))
        for value, power in zip(coefficients, exponent, strict=True):
            coefficient *= value**power
        if coefficient:
            result[exponent] = coefficient
    return result


def _nested_decomposition(
    order: int, depth: int, target_word: Word
) -> tuple[Polynomial, list[Fraction]]:
    if target_word:
        row, column, degree = _embedding_coordinate(target_word)
    else:
        row, column, degree = 0, 0, 0
    outer_weights = _coefficient_weights(order, depth, degree)
    inner_weights = _mat2_weights(order, row, column)
    grid = _simplex(order)
    observed: Polynomial = {}
    term_weights = []
    for node, outer_weight in enumerate(outer_weights):
        for point, inner_weight in zip(grid, inner_weights, strict=True):
            weight = outer_weight * inner_weight
            if not weight:
                continue
            coefficients = _embedded_linear_form(depth, node, point)
            if not any(coefficients):
                continue
            term_weights.append(weight)
            observed = _add(
                observed,
                _scale(_linear_form_power(order, coefficients), weight),
            )
    return observed, term_weights


def _fraction_payload(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_certificate(
    max_order: int = 3, max_depth: int = 4
) -> dict[str, object]:
    cases = []
    for order in range(1, max_order + 1):
        for depth in range(1, max_depth + 1):
            direct = _direct_power(order, depth)
            term_counts = []
            term_hashes = []
            target_hashes = []
            all_weights = []
            for word in _words(depth):
                observed, weights = _nested_decomposition(order, depth, word)
                expected = direct.get(word, {})
                if observed != expected:
                    raise AssertionError(
                        "alternating-word collapse failed for "
                        f"M={order}, d={depth}, boundary={_word_payload(word)}"
                    )
                term_counts.append(len(weights))
                all_weights.extend(weights)
                term_hashes.append(
                    _digest([_fraction_payload(weight) for weight in weights])
                )
                target_hashes.append(
                    _digest(
                        [
                            [*exponent, _fraction_payload(value)]
                            for exponent, value in sorted(expected.items())
                        ]
                    )
                )
            cases.append(
                {
                    "pairs": order,
                    "radical_nilpotency_index": depth,
                    "algebra_dimension": 2 * depth - 1,
                    "lc_agp_term_bound": (
                        (order * (depth - 1) + 1)
                        * math.comb(order + 3, 3)
                    ),
                    "verified_boundary_words": [
                        _word_payload(word) for word in _words(depth)
                    ],
                    "nonzero_terms_by_boundary": term_counts,
                    "maximum_weight_numerator_bits": max(
                        abs(weight.numerator).bit_length()
                        for weight in all_weights
                    ),
                    "maximum_weight_denominator_bits": max(
                        weight.denominator.bit_length()
                        for weight in all_weights
                    ),
                    "term_weights_sha256": _digest(term_hashes),
                    "targets_sha256": _digest(target_hashes),
                }
            )
    body: dict[str, object] = {
        "schema": "femps.alternating-word-pair-lc-agp-collapse.v1",
        "arithmetic": "exact rational",
        "algebra": "Q<x,y>/(x^2,y^2,words of length d)",
        "embedding": "w -> z^len(w) E_(first(w),complement(last(w)))",
        "boundary": "every alternating-word basis functional",
        "term_bound": "[M(d-1)+1] binom(M+3,3)",
        "max_pairs": max_order,
        "max_radical_nilpotency_index": max_depth,
        "cases": cases,
    }
    body["certificate_sha256"] = _digest(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.verify is not None and arguments.output is not None:
        raise SystemExit("choose either --verify or --output")
    if arguments.max_order < 1 or arguments.max_depth < 1:
        raise SystemExit("max-order and max-depth must be positive")
    observed = build_certificate(arguments.max_order, arguments.max_depth)
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
