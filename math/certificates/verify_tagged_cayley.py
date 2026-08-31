"""Independent exact verifier for the tagged Cayley-determinant reduction.

This script deliberately imports neither PyTorch nor ``femps``.  It enumerates
perfect matchings and all product orders using sparse integer matrices, then
checks the scaled top-form coefficient against a row-ordered Cayley
determinant.  The arithmetic is exact over the integers.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import TypeAlias

SparseMatrix: TypeAlias = dict[tuple[int, int], int]


def _add(first: SparseMatrix, second: SparseMatrix, scale: int = 1) -> SparseMatrix:
    result = dict(first)
    for key, value in second.items():
        updated = result.get(key, 0) + scale * value
        if updated:
            result[key] = updated
        else:
            result.pop(key, None)
    return result


def _matmul(first: SparseMatrix, second: SparseMatrix) -> SparseMatrix:
    if not first or not second:
        return {}
    by_row: dict[int, list[tuple[int, int]]] = {}
    for (row, column), value in second.items():
        by_row.setdefault(row, []).append((column, value))
    result: SparseMatrix = {}
    for (row, middle), first_value in first.items():
        for column, second_value in by_row.get(middle, ()):
            key = (row, column)
            result[key] = result.get(key, 0) + first_value * second_value
    return {key: value for key, value in result.items() if value}


def _identity(order: int) -> SparseMatrix:
    return {(index, index): 1 for index in range(order)}


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[first] > permutation[second]
        for first in range(len(permutation))
        for second in range(first + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _perfect_matchings(
    indices: tuple[int, ...],
) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    if not indices:
        return [(1, ())]
    first = indices[0]
    terms = []
    for offset, second in enumerate(indices[1:], start=1):
        remaining = indices[1:offset] + indices[offset + 1 :]
        local_sign = -1 if (offset + 1) % 2 else 1
        for nested_sign, nested_pairs in _perfect_matchings(remaining):
            terms.append((local_sign * nested_sign, ((first, second),) + nested_pairs))
    return terms


def _deterministic_entries(order: int, block: int = 2) -> list[list[list[list[int]]]]:
    return [
        [
            [
                [
                    ((97 * order + 31 * row + 17 * column + 7 * inner_row + 3 * inner_column) % 11)
                    - 5
                    for inner_column in range(block)
                ]
                for inner_row in range(block)
            ]
            for column in range(order)
        ]
        for row in range(order)
    ]


def _block_matrix(block_values: list[list[int]]) -> SparseMatrix:
    return {
        (row, column): value
        for row, values in enumerate(block_values)
        for column, value in enumerate(values)
        if value
    }


def _cayley_determinant(entries: list[list[list[list[int]]]]) -> SparseMatrix:
    order = len(entries)
    block = len(entries[0][0])
    total: SparseMatrix = {}
    for permutation in itertools.permutations(range(order)):
        product = _identity(block)
        for row, column in enumerate(permutation):
            product = _matmul(product, _block_matrix(entries[row][column]))
        total = _add(total, product, _permutation_sign(permutation))
    return total


def _tagged_pair_matrices(
    entries: list[list[list[list[int]]]],
) -> tuple[dict[tuple[int, int], SparseMatrix], int]:
    order = len(entries)
    block = len(entries[0][0])
    virtual_order = (order + 1) * block
    matrices: dict[tuple[int, int], SparseMatrix] = {}
    for row in range(order):
        for column in range(order):
            tagged: SparseMatrix = {}
            for left in range(block):
                for right in range(block):
                    value = entries[row][column][left][right]
                    if value:
                        tagged[(row * block + left, (row + 1) * block + right)] = value
            matrices[(row, order + column)] = tagged
    return matrices, virtual_order


def _scaled_symmetrized_pfaffian(
    pair_matrices: dict[tuple[int, int], SparseMatrix],
    physical_order: int,
    virtual_order: int,
) -> SparseMatrix:
    pairs = physical_order // 2
    total: SparseMatrix = {}
    for matching_sign, matching in _perfect_matchings(tuple(range(physical_order))):
        factors = [pair_matrices.get(pair, {}) for pair in matching]
        for order_of_factors in itertools.permutations(range(pairs)):
            product = _identity(virtual_order)
            for factor_index in order_of_factors:
                product = _matmul(product, factors[factor_index])
            total = _add(total, product, matching_sign)
    return total


def _expected_tagged_cayley(
    cayley: SparseMatrix, order: int, block: int
) -> SparseMatrix:
    phase = -1 if (order * (order - 1) // 2) % 2 else 1
    return {
        (left, order * block + right): phase * value
        for (left, right), value in cayley.items()
        if value
    }


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _matrix_payload(matrix: SparseMatrix) -> list[list[int]]:
    return [[row, column, value] for (row, column), value in sorted(matrix.items())]


def build_certificate(max_order: int = 4) -> dict[str, object]:
    cases = []
    for order in range(1, max_order + 1):
        entries = _deterministic_entries(order)
        pair_matrices, virtual_order = _tagged_pair_matrices(entries)
        observed = _scaled_symmetrized_pfaffian(
            pair_matrices, 2 * order, virtual_order
        )
        cayley = _cayley_determinant(entries)
        expected = _expected_tagged_cayley(cayley, order, 2)
        if observed != expected:
            raise AssertionError(f"tagged identity failed at order {order}")
        cases.append(
            {
                "order": order,
                "block_dimension": 2,
                "physical_dimension": 2 * order,
                "virtual_dimension": 2 * (order + 1),
                "matching_count": math.prod(range(1, 2 * order, 2)),
                "product_order_count": math.factorial(order),
                "input_sha256": _digest(entries),
                "scaled_coefficient_sha256": _digest(_matrix_payload(observed)),
                "nonzero_scaled_entries": len(observed),
            }
        )
    body: dict[str, object] = {
        "schema": "femps.tagged-cayley-certificate.v1",
        "arithmetic": "exact integers",
        "enumerator": "perfect matchings times all factor permutations",
        "orders": list(range(1, max_order + 1)),
        "cases": cases,
    }
    body["certificate_sha256"] = _digest(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--max-order", type=int, default=4)
    arguments = parser.parse_args()
    observed = build_certificate(arguments.max_order)
    if arguments.verify is not None:
        expected = json.loads(arguments.verify.read_text(encoding="utf-8"))
        if observed != expected:
            raise SystemExit("certificate mismatch")
        print(f"verified {arguments.verify} ({observed['certificate_sha256']})")
        return
    print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
