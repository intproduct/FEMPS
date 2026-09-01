"""Independently verify the seven-dimensional four-form orbit-rank table.

This standard-library-only verifier imports neither ``exact_contractions`` nor
the FEMPS package.  It checks the certificate payload and exact calculations.
Cohen--Helminck Theorem 2.1 remains the separate source-backed assertion that
the nine recorded characteristic-zero orbit representatives are exhaustive.
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


def _canonical_terms(raw_terms: object) -> dict[tuple[int, ...], Fraction]:
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError("three_form_terms must be a nonempty list")
    terms: dict[tuple[int, ...], Fraction] = {}
    for raw in raw_terms:
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("each source term must be a three-index list")
        indices = tuple(index - 1 for index in raw)
        if not all(isinstance(index, int) and 0 <= index < 7 for index in indices):
            raise ValueError("source indices must lie in 1,...,7")
        sign = _sign(indices)
        if sign == 0:
            raise ValueError("source term contains a repeated index")
        key = tuple(sorted(indices))
        terms[key] = terms.get(key, Fraction(0)) + sign
        if terms[key] == 0:
            del terms[key]
    if not terms:
        raise ValueError("source terms cancel to zero")
    return terms


def _hodge_dual(
    terms: dict[tuple[int, ...], Fraction], dimension: int
) -> dict[tuple[int, ...], Fraction]:
    ambient = set(range(dimension))
    result: dict[tuple[int, ...], Fraction] = {}
    for key, coefficient in terms.items():
        complement = tuple(sorted(ambient.difference(key)))
        result[complement] = coefficient * _sign(key + complement)
    return result


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


def _rank(matrix: list[list[Fraction]]) -> int:
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


def verify(path: Path) -> dict[str, object]:
    certificate = json.loads(path.read_text(encoding="utf-8"))
    if certificate.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    if certificate.get("ambient_dimension") != 7:
        raise ValueError("expected ambient dimension seven")
    source = certificate.get("classification_source")
    if not isinstance(source, dict) or source.get("doi") != "10.1080/00927878808823558":
        raise ValueError("classification source is missing or changed")

    recorded_hash = certificate.get("payload_sha256")
    payload = {key: value for key, value in certificate.items() if key != "payload_sha256"}
    if recorded_hash != _digest(payload):
        raise ValueError("payload_sha256 mismatch")

    raw_orbits = certificate.get("orbits")
    if not isinstance(raw_orbits, list) or len(raw_orbits) != 9:
        raise ValueError("expected exactly nine source orbit representatives")
    expected_names = [f"f{index}" for index in range(1, 10)]
    if [orbit.get("name") for orbit in raw_orbits if isinstance(orbit, dict)] != expected_names:
        raise ValueError("orbit names or ordering do not match f1,...,f9")

    verified: list[dict[str, object]] = []
    for orbit in raw_orbits:
        if not isinstance(orbit, dict):
            raise ValueError("each orbit must be an object")
        trivector = _canonical_terms(orbit.get("three_form_terms"))
        four_form = _hodge_dual(trivector, 7)
        matrices = [_matrix(four_form, 7, degree) for degree in range(5)]
        ranks = [_rank(matrix) for matrix in matrices]
        if ranks != orbit.get("contraction_ranks"):
            raise ValueError(f"rank mismatch for {orbit.get('name')}")
        concise = ranks[1] == 7
        if concise is not orbit.get("concise"):
            raise ValueError(f"conciseness mismatch for {orbit.get('name')}")
        if matrices[2] != [list(row) for row in zip(*matrices[2], strict=True)]:
            raise ValueError(f"middle contraction is not symmetric for {orbit.get('name')}")
        verified.append({"name": orbit["name"], "ranks": ranks, "concise": concise})

    minimum = min(item["ranks"][2] for item in verified if item["concise"])
    conclusion = certificate.get("conclusion")
    if not isinstance(conclusion, dict) or conclusion.get("value") != minimum:
        raise ValueError("recorded minimum does not match concise orbit table")
    if minimum != 12 or conclusion.get("rational_witness_orbit") != "f3":
        raise ValueError("unexpected seven-dimensional conclusion")

    f3_terms = _hodge_dual(_canonical_terms(raw_orbits[2]["three_form_terms"]), 7)
    recorded_witness = {
        tuple(index - 1 for index in term["indices"]): Fraction(term["coefficient"])
        for term in conclusion.get("rational_witness_four_form_terms", [])
    }
    if f3_terms != recorded_witness:
        raise ValueError("recorded rational witness is not the dual of f3")

    return {
        "artifact": str(path),
        "ambient_dimension": 7,
        "verified_orbits": verified,
        "concise_middle_rank_minimum": minimum,
        "classification_exhaustiveness": "source-backed by Cohen--Helminck Theorem 2.1",
        "payload_sha256": recorded_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.verify), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
