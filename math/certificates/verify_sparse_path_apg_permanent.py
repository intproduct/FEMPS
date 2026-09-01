"""Exact certificate for the sparse-path APG permanent reduction.

The verifier deliberately imports neither PyTorch nor ``femps``.  It compares
three exact-integer routes:

1. propagation through an upper-bidiagonal virtual path whose edges carry
   pair forms;
2. direct exterior multiplication in a square-zero commuting pair basis; and
3. the permutation definition of the permanent.

The physical pair basis is ``P_j=e_(2j) wedge e_(2j+1)``.  Its elements have
even exterior degree, hence commute, while ``P_j wedge P_j=0``.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path


Matrix = list[list[int]]
ExteriorVector = dict[int, int]


def _permanent(matrix: Matrix) -> int:
    order = len(matrix)
    return sum(
        math.prod(matrix[row][column] for row, column in enumerate(permutation))
        for permutation in itertools.permutations(range(order))
    )


def _wedge_pair_vectors(first: ExteriorVector, second: ExteriorVector) -> ExteriorVector:
    """Multiply square-zero commuting pair monomials represented by bit masks."""

    result: ExteriorVector = {}
    for first_mask, first_value in first.items():
        for second_mask, second_value in second.items():
            if first_mask & second_mask:
                continue
            mask = first_mask | second_mask
            result[mask] = result.get(mask, 0) + first_value * second_value
    return {mask: value for mask, value in result.items() if value}


def _edge_form(row: list[int]) -> ExteriorVector:
    return {1 << column: value for column, value in enumerate(row) if value}


def _direct_exterior_product(matrix: Matrix) -> ExteriorVector:
    value: ExteriorVector = {0: 1}
    for row in matrix:
        value = _wedge_pair_vectors(value, _edge_form(row))
    return value


def _upper_bidiagonal_path_power(matrix: Matrix) -> dict[int, ExteriorVector]:
    """Propagate ``e_0^T Omega^M`` without shortcutting the virtual path."""

    order = len(matrix)
    state: dict[int, ExteriorVector] = {0: {0: 1}}
    for _ in range(order):
        updated: dict[int, ExteriorVector] = {}
        for source, coefficient in state.items():
            if source >= order:
                continue
            target = source + 1
            contribution = _wedge_pair_vectors(
                coefficient, _edge_form(matrix[source])
            )
            if target not in updated:
                updated[target] = contribution
            else:
                for mask, value in contribution.items():
                    updated[target][mask] = updated[target].get(mask, 0) + value
        state = updated
    return state


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _deterministic_binary(order: int) -> Matrix:
    return [
        [
            int((17 * order + 11 * row + 7 * column + 3 * row * column) % 7 < 3)
            for column in range(order)
        ]
        for row in range(order)
    ]


def _matrices(order: int) -> list[tuple[str, Matrix]]:
    identity = [[int(row == column) for column in range(order)] for row in range(order)]
    all_ones = [[1 for _ in range(order)] for _ in range(order)]
    return [
        ("identity", identity),
        ("all_ones", all_ones),
        ("deterministic_binary", _deterministic_binary(order)),
    ]


def build_certificate(max_order: int = 6) -> dict[str, object]:
    cases = []
    for order in range(1, max_order + 1):
        top_mask = (1 << order) - 1
        for label, matrix in _matrices(order):
            permanent = _permanent(matrix)
            exterior = _direct_exterior_product(matrix)
            path = _upper_bidiagonal_path_power(matrix)
            path_endpoint = path.get(order, {})
            expected = {top_mask: permanent} if permanent else {}
            if exterior != expected:
                raise AssertionError(
                    f"exterior/permanent mismatch for M={order}, {label}"
                )
            if path_endpoint != expected or set(path) != {order}:
                raise AssertionError(
                    f"path/exterior mismatch for M={order}, {label}"
                )
            normalized_norm = Fraction(permanent * permanent, math.factorial(order) ** 2)
            cases.append(
                {
                    "pairs": order,
                    "matrix_family": label,
                    "matrix_sha256": _digest(matrix),
                    "permanent": permanent,
                    "top_pair_mask": top_mask,
                    "normalized_norm_squared": (
                        f"{normalized_norm.numerator}/{normalized_norm.denominator}"
                    ),
                    "path_virtual_width": order + 1,
                    "physical_one_particle_dimension": 2 * order,
                }
            )
    body: dict[str, object] = {
        "schema": "femps.sparse-path-apg-permanent.v1",
        "arithmetic": "exact integers and rationals",
        "virtual_generator": "upper bidiagonal, Omega_(i,i+1)=sum_j A_(i,j) P_j",
        "pair_algebra": "P_j P_k=P_k P_j and P_j^2=0",
        "state_convention": "e_0^T Omega^M e_M / M!",
        "verified_identity": (
            "state=perm(A) P_1...P_M/M!, norm2=perm(A)^2/(M!)^2"
        ),
        "independent_routes": [
            "upper-bidiagonal virtual-path propagation",
            "square-zero commuting exterior subset propagation",
            "permutation enumeration",
        ],
        "max_pairs": max_order,
        "cases": cases,
    }
    body["certificate_sha256"] = _digest(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--max-order", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.max_order < 1:
        raise SystemExit("max-order must be positive")
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
