"""Independent exact verifier for the fixed-bond Cayley FEMPS reduction.

The verifier imports neither PyTorch nor FEMPS. It compares a direct
row-ordered matrix determinant with an explicit sum over one-form FEMPS
virtual paths and physical permutations. It also checks exact norm
polarization after adjoining one scalar reference path, which raises the
maximum virtual bond from two to three.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


Matrix = list[list[int]]
EntryArray = list[list[Matrix]]


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    size = len(left)
    return [
        [
            sum(
                left[row][inner] * right[inner][column]
                for inner in range(size)
            )
            for column in range(size)
        ]
        for row in range(size)
    ]


def _add(left: Matrix, right: Matrix, scale: int = 1) -> Matrix:
    return [
        [
            left[row][column] + scale * right[row][column]
            for column in range(len(left))
        ]
        for row in range(len(left))
    ]


def _identity(size: int) -> Matrix:
    return [
        [int(row == column) for column in range(size)] for row in range(size)
    ]


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[first] > permutation[second]
        for first in range(len(permutation))
        for second in range(first + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _deterministic_entries(order: int) -> EntryArray:
    return [
        [
            [
                [
                    (
                        37 * (row + 1)
                        + 19 * (column + 1)
                        + 11 * (left + 1)
                        + 7 * (right + 1)
                        + (row + 1)
                        * (column + 1)
                        * (left + 2)
                        * (right + 3)
                    )
                    % 13
                    - 6
                    for right in range(2)
                ]
                for left in range(2)
            ]
            for column in range(order)
        ]
        for row in range(order)
    ]


def _cayley_determinant(entries: EntryArray) -> Matrix:
    order = len(entries)
    total = [[0, 0], [0, 0]]
    for permutation in itertools.permutations(range(order)):
        product = _identity(2)
        for row, column in enumerate(permutation):
            product = _matmul(product, entries[row][column])
        total = _add(total, product, _permutation_sign(permutation))
    return total


def _path_amplitude(
    entries: EntryArray,
    permutation: tuple[int, ...],
    left_boundary: list[int],
    right_boundary: list[int],
) -> int:
    order = len(entries)
    bond = len(left_boundary)
    total = 0
    left_states = [index for index, value in enumerate(left_boundary) if value]
    right_states = [index for index, value in enumerate(right_boundary) if value]
    for left, right in itertools.product(left_states, right_states):
        for internal in itertools.product(range(bond), repeat=max(0, order - 1)):
            path = (left, *internal, right)
            product = left_boundary[left] * right_boundary[right]
            for site, physical in enumerate(permutation):
                product *= entries[site][physical][path[site]][path[site + 1]]
            total += product
    return _permutation_sign(permutation) * total


def _direct_femps_top_amplitude(
    entries: EntryArray,
    left_boundary: list[int],
    right_boundary: list[int],
) -> int:
    return sum(
        _path_amplitude(entries, permutation, left_boundary, right_boundary)
        for permutation in itertools.permutations(range(len(entries)))
    )


def _basis_boundary(index: int, size: int) -> list[int]:
    return [int(position == index) for position in range(size)]


def _polarized_entries(entries: EntryArray) -> EntryArray:
    order = len(entries)
    polarized: EntryArray = []
    for site in range(order):
        row_entries = []
        for physical in range(order):
            block = [[0 for _ in range(3)] for _ in range(3)]
            for left, right in itertools.product(range(2), repeat=2):
                block[left][right] = entries[site][physical][left][right]
            block[2][2] = int(site == physical)
            row_entries.append(block)
        polarized.append(row_entries)
    return polarized


def build_certificate(min_order: int = 2, max_order: int = 6) -> dict[str, object]:
    if min_order < 2 or max_order < min_order:
        raise ValueError("require 2 <= min_order <= max_order")
    cases = []
    for order in range(min_order, max_order + 1):
        entries = _deterministic_entries(order)
        cayley = _cayley_determinant(entries)
        polarized_entries = _polarized_entries(entries)
        amplitudes = []
        for left, right in itertools.product(range(2), repeat=2):
            expected = cayley[left][right]
            observed = _direct_femps_top_amplitude(
                entries,
                _basis_boundary(left, 2),
                _basis_boundary(right, 2),
            )
            if observed != expected:
                raise AssertionError(
                    f"fixed-bond Cayley identity failed at n={order}, ({left},{right})"
                )
            hard_squared_norm = observed * observed
            polarized_left = _basis_boundary(left, 3)
            polarized_right = _basis_boundary(right, 3)
            polarized_left[2] = 1
            polarized_right[2] = 1
            shifted_observed = _direct_femps_top_amplitude(
                polarized_entries, polarized_left, polarized_right
            )
            if shifted_observed != observed + 1:
                raise AssertionError(
                    f"direct-sum amplitude failed at n={order}, ({left},{right})"
                )
            shifted_squared_norm = shifted_observed * shifted_observed
            recovered = (shifted_squared_norm - hard_squared_norm - 1) // 2
            if recovered != observed:
                raise AssertionError(
                    f"norm polarization failed at n={order}, ({left},{right})"
                )
            amplitudes.append(
                {
                    "left": left,
                    "right": right,
                    "top_exterior_coefficient": observed,
                    "hard_squared_norm": hard_squared_norm,
                    "reference_coefficient": 1,
                    "polarized_top_exterior_coefficient": shifted_observed,
                    "shifted_squared_norm": shifted_squared_norm,
                    "polarization_recovery": recovered,
                }
            )
        cases.append(
            {
                "order": order,
                "one_particle_dimension": order,
                "matrix_block_dimension": 2,
                "hard_femps_max_bond": 2,
                "polarized_femps_max_bond": 3,
                "physical_permutation_count": math.factorial(order),
                "input_sha256": _digest(entries),
                "cayley_matrix": cayley,
                "amplitudes": amplitudes,
                "antisymmetry_residual": 0,
            }
        )
    body: dict[str, object] = {
        "schema": "femps.fixed-bond-cayley-certificate.v1",
        "arithmetic": "exact integers",
        "construction": "site-indexed one-form cores with direct Cayley coefficient",
        "reduction": "two squared-norm queries and a scalar reference direct sum",
        "min_order": min_order,
        "max_order": max_order,
        "cases": cases,
    }
    body["certificate_sha256"] = _digest(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--min-order", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=6)
    arguments = parser.parse_args()
    observed = build_certificate(arguments.min_order, arguments.max_order)
    if arguments.verify is not None:
        expected = json.loads(arguments.verify.read_text(encoding="utf-8"))
        if observed != expected:
            raise SystemExit("certificate mismatch")
        print(f"verified {arguments.verify} ({observed['certificate_sha256']})")
        return
    print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
