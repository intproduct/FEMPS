"""Independent verifier for the exact 16D direct-sum four-form control.

The script imports neither ``exact_contractions`` nor any numerical package.
It verifies a rational certificate by rebuilding all five contraction maps,
performing exact Fraction Gaussian elimination, and checking the middle-map
symmetry and complementary-map transpose convention.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sign(indices: tuple[int, ...]) -> int:
    if len(set(indices)) != len(indices):
        return 0
    inversions = sum(
        indices[left] > indices[right]
        for left in range(len(indices))
        for right in range(left + 1, len(indices))
    )
    return -1 if inversions % 2 else 1


def _basis(dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.combinations(range(dimension), degree))


def _rank(matrix: list[list[Fraction]]) -> int:
    if not matrix:
        return 0
    work = [row[:] for row in matrix]
    pivot_row = 0
    for column in range(len(work[0])):
        pivot = next(
            (row for row in range(pivot_row, len(work)) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        pivot_value = work[pivot_row][column]
        work[pivot_row] = [entry / pivot_value for entry in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or work[row][column] == 0:
                continue
            factor = work[row][column]
            work[row] = [
                entry - factor * pivot_entry
                for entry, pivot_entry in zip(work[row], work[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def _matrix(
    terms: dict[tuple[int, ...], Fraction], dimension: int, input_degree: int
) -> list[list[Fraction]]:
    columns = _basis(dimension, input_degree)
    rows = _basis(dimension, 4 - input_degree)
    return [
        [
            _sign(left + right)
            * terms.get(tuple(sorted(left + right)), Fraction(0))
            for left in columns
        ]
        for right in rows
    ]


def _load_terms(certificate: dict[str, object]) -> dict[tuple[int, ...], Fraction]:
    raw_terms = certificate.get("form_terms")
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError("form_terms must be a nonempty list")
    terms: dict[tuple[int, ...], Fraction] = {}
    dimension = certificate["ambient_dimension"]
    if not isinstance(dimension, int):
        raise ValueError("ambient_dimension must be an integer")
    for raw in raw_terms:
        if not isinstance(raw, dict):
            raise ValueError("each form term must be an object")
        indices = tuple(raw["indices"])
        if len(indices) != 4 or tuple(sorted(indices)) != indices:
            raise ValueError("term indices must be increasing 4-tuples")
        if len(set(indices)) != 4 or not all(0 <= index < dimension for index in indices):
            raise ValueError("term indices must be distinct ambient indices")
        coefficient = Fraction(raw["numerator"], raw["denominator"])
        if coefficient == 0 or indices in terms:
            raise ValueError("terms must be nonzero and unique")
        terms[indices] = coefficient
    return terms


def verify(path: Path) -> dict[str, object]:
    certificate = json.loads(path.read_text(encoding="utf-8"))
    if certificate.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    if certificate.get("claim_status") != "exact certificate":
        raise ValueError("control must be labeled exact certificate")
    if certificate.get("base_field") != {"kind": "Q", "characteristic": 0}:
        raise ValueError("this verifier accepts only the rational base field")
    if certificate.get("form_degree") != 4:
        raise ValueError("expected a four-form")
    dimension = certificate.get("ambient_dimension")
    if dimension != 16:
        raise ValueError("expected the 16-dimensional control")

    recorded_hash = certificate.get("payload_sha256")
    payload = {key: value for key, value in certificate.items() if key != "payload_sha256"}
    if recorded_hash != _digest(payload):
        raise ValueError("payload_sha256 mismatch")

    terms = _load_terms(certificate)
    matrices = [_matrix(terms, dimension, degree) for degree in range(5)]
    ranks = [_rank(matrix) for matrix in matrices]
    if ranks != certificate.get("contraction_ranks"):
        raise ValueError("recorded contraction ranks do not match exact elimination")
    if ranks != [1, 16, 24, 16, 1]:
        raise ValueError("unexpected direct-sum control rank vector")

    middle = matrices[2]
    if middle != [list(row) for row in zip(*middle, strict=True)]:
        raise ValueError("middle contraction is not symmetric")
    first = matrices[1]
    third = matrices[3]
    signed_transpose = [
        [-first[column][row] for column in range(len(first))]
        for row in range(len(first[0]))
    ]
    if third != signed_transpose:
        raise ValueError("C_3 is not -C_1^T in the recorded convention")

    block_supports = [set(term) for term in terms]
    if len(block_supports) != 4 or set().union(*block_supports) != set(range(16)):
        raise ValueError("control does not cover four 4D ambient blocks")
    if any(left & right for index, left in enumerate(block_supports) for right in block_supports[index + 1 :]):
        raise ValueError("control blocks are not disjoint")

    return {
        "artifact": str(path),
        "base_field": "Q",
        "ambient_dimension": dimension,
        "contraction_ranks": ranks,
        "concise": ranks[1] == dimension,
        "payload_sha256": recorded_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.verify), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
