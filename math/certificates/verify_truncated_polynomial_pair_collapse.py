"""Exact verifier for the C[z]/(z^d) matrix-pair LC-AGP collapse.

For every boundary basis functional [z^s], the verifier constructs exact
rational interpolation weights on integer nodes, expands both sides as
homogeneous polynomials in the scalar two-form coordinates, and compares every
coefficient.  Boundary-basis verification implies arbitrary-boundary coverage
by linearity.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator, TypeAlias

Exponent = tuple[int, ...]
Polynomial: TypeAlias = dict[Exponent, Fraction]


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


def _weighted_degree(exponent: Exponent) -> int:
    return sum(index * power for index, power in enumerate(exponent))


def _target(order: int, depth: int, boundary_degree: int) -> Polynomial:
    return {
        exponent: Fraction(_multinomial(order, exponent))
        for exponent in _exponents(order, depth)
        if _weighted_degree(exponent) == boundary_degree
    }


def _evaluated_power(order: int, depth: int, node: int) -> Polynomial:
    return {
        exponent: Fraction(
            _multinomial(order, exponent)
            * node ** _weighted_degree(exponent)
        )
        for exponent in _exponents(order, depth)
        if node or _weighted_degree(exponent) == 0
    }


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


def _coefficient_weights(
    order: int, depth: int, boundary_degree: int
) -> list[Fraction]:
    term_bound = order * (depth - 1) + 1
    nodes = range(term_bound)
    vandermonde_transpose = [
        [Fraction(node**degree) for node in nodes]
        for degree in range(term_bound)
    ]
    right_hand_side = [
        Fraction(degree == boundary_degree)
        for degree in range(term_bound)
    ]
    return _fraction_solve(vandermonde_transpose, right_hand_side)


def _reconstruct(
    order: int, depth: int, weights: list[Fraction]
) -> Polynomial:
    observed: Polynomial = {}
    for node, weight in enumerate(weights):
        for exponent, coefficient in _evaluated_power(
            order, depth, node
        ).items():
            value = observed.get(exponent, Fraction()) + weight * coefficient
            if value:
                observed[exponent] = value
            else:
                observed.pop(exponent, None)
    return observed


def _fraction_payload(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_certificate(
    max_order: int = 4, max_depth: int = 4
) -> dict[str, object]:
    cases = []
    for order in range(1, max_order + 1):
        for depth in range(1, max_depth + 1):
            weights_by_boundary = []
            targets = []
            for boundary_degree in range(depth):
                weights = _coefficient_weights(
                    order, depth, boundary_degree
                )
                target = _target(order, depth, boundary_degree)
                if _reconstruct(order, depth, weights) != target:
                    raise AssertionError(
                        "truncated-polynomial collapse failed for "
                        f"M={order}, d={depth}, s={boundary_degree}"
                    )
                weights_by_boundary.append(weights)
                targets.append(target)
            nonzero_weights = [
                weight
                for weights in weights_by_boundary
                for weight in weights
                if weight
            ]
            cases.append(
                {
                    "pairs": order,
                    "radical_nilpotency_index": depth,
                    "lc_agp_term_bound": order * (depth - 1) + 1,
                    "verified_boundary_basis": list(range(depth)),
                    "nonzero_terms_by_boundary": [
                        sum(bool(weight) for weight in weights)
                        for weights in weights_by_boundary
                    ],
                    "maximum_weight_numerator_bits": max(
                        abs(weight.numerator).bit_length()
                        for weight in nonzero_weights
                    ),
                    "maximum_weight_denominator_bits": max(
                        weight.denominator.bit_length()
                        for weight in nonzero_weights
                    ),
                    "weights_sha256": _digest(
                        [
                            [_fraction_payload(weight) for weight in weights]
                            for weights in weights_by_boundary
                        ]
                    ),
                    "targets_sha256": _digest(
                        [
                            [
                                [*exponent, _fraction_payload(value)]
                                for exponent, value in sorted(target.items())
                            ]
                            for target in targets
                        ]
                    ),
                }
            )
    body: dict[str, object] = {
        "schema": "femps.truncated-polynomial-pair-lc-agp-collapse.v1",
        "arithmetic": "exact rational",
        "algebra": "Q[z]/(z^d)",
        "generator": "Omega(z)=sum_(j=0)^(d-1) F_j z^j",
        "boundary": "arbitrary linear functional on 1,z,...,z^(d-1)",
        "term_bound": "M(d-1)+1",
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
    parser.add_argument("--max-order", type=int, default=4)
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
