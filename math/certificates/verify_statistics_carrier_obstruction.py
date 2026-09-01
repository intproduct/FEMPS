"""Exact certificate for the statistics-carrier dimension obstruction.

For every tested particle number ``N``, the verifier compares a Slater
determinant with

    e1 wedge (e2 wedge e3 + e4 wedge e5) wedge e6 ... wedge e_(N+2).

All exterior flattenings and ranks are computed over the rationals without
NumPy, SymPy, PyTorch, or FEMPS imports.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path


ExteriorForm = dict[tuple[int, ...], int]
Matrix = list[list[Fraction]]


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _shuffle_sign(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    inversions = sum(first > second for first in left for second in right)
    return -1 if inversions % 2 else 1


def _sequence_sign(sequence: tuple[int, ...]) -> int:
    inversions = sum(
        sequence[first] > sequence[second]
        for first in range(len(sequence))
        for second in range(first + 1, len(sequence))
    )
    return -1 if inversions % 2 else 1


def _permute_form(form: ExteriorForm, permutation: list[int]) -> ExteriorForm:
    transformed: ExteriorForm = {}
    for support, coefficient in form.items():
        mapped = tuple(permutation[index] for index in support)
        ordered = tuple(sorted(mapped))
        transformed[ordered] = (
            transformed.get(ordered, 0) + coefficient * _sequence_sign(mapped)
        )
    return {support: value for support, value in transformed.items() if value}


def _flattening(
    form: ExteriorForm,
    *,
    dimension: int,
    particles: int,
    cut: int,
) -> Matrix:
    left_supports = list(itertools.combinations(range(dimension), cut))
    right_supports = list(
        itertools.combinations(range(dimension), particles - cut)
    )
    left_index = {support: index for index, support in enumerate(left_supports)}
    right_index = {
        support: index for index, support in enumerate(right_supports)
    }
    matrix = [
        [Fraction(0) for _ in right_supports] for _ in left_supports
    ]
    for support, coefficient in form.items():
        for left in itertools.combinations(support, cut):
            left_set = set(left)
            right = tuple(index for index in support if index not in left_set)
            matrix[left_index[left]][right_index[right]] += (
                coefficient * _shuffle_sign(left, right)
            )
    return matrix


def _rank(matrix: Matrix) -> int:
    if not matrix:
        return 0
    reduced = [row.copy() for row in matrix]
    rows = len(reduced)
    columns = len(reduced[0])
    pivot_row = 0
    for column in range(columns):
        pivot = next(
            (
                row
                for row in range(pivot_row, rows)
                if reduced[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        reduced[pivot_row], reduced[pivot] = reduced[pivot], reduced[pivot_row]
        pivot_value = reduced[pivot_row][column]
        reduced[pivot_row] = [value / pivot_value for value in reduced[pivot_row]]
        for row in range(rows):
            if row == pivot_row or not reduced[row][column]:
                continue
            factor = reduced[row][column]
            reduced[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(
                    reduced[row], reduced[pivot_row], strict=True
                )
            ]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def _slater(support: tuple[int, ...]) -> ExteriorForm:
    return {support: 1}


def _counterexample(particles: int) -> tuple[ExteriorForm, ExteriorForm, ExteriorForm]:
    shared_tail = tuple(range(5, particles + 2))
    first_support = (0, 1, 2, *shared_tail)
    second_support = (0, 3, 4, *shared_tail)
    first = _slater(first_support)
    second = _slater(second_support)
    total = {**first, **second}
    return first, second, total


def _cut_ranks(
    form: ExteriorForm, *, dimension: int, particles: int
) -> list[int]:
    return [
        _rank(
            _flattening(
                form,
                dimension=dimension,
                particles=particles,
                cut=cut,
            )
        )
        for cut in range(1, particles)
    ]


def _row_space_sum_rank(first: Matrix, second: Matrix) -> int:
    return _rank([*first, *second])


def build_certificate(min_particles: int = 3, max_particles: int = 8) -> dict[str, object]:
    if min_particles < 3 or max_particles < min_particles:
        raise ValueError("require 3 <= min_particles <= max_particles")
    cases = []
    for particles in range(min_particles, max_particles + 1):
        dimension = particles + 2
        slater = _slater(tuple(range(particles)))
        first, second, counterexample = _counterexample(particles)
        slater_ranks = _cut_ranks(
            slater, dimension=dimension, particles=particles
        )
        expected_slater_ranks = [
            math.comb(particles, cut) for cut in range(1, particles)
        ]
        if slater_ranks != expected_slater_ranks:
            raise AssertionError(f"Slater ranks failed for N={particles}")

        counterexample_ranks = _cut_ranks(
            counterexample, dimension=dimension, particles=particles
        )
        if counterexample_ranks[0] != particles + 2:
            raise AssertionError(f"counterexample rank failed for N={particles}")
        if counterexample_ranks[0] % particles == 0:
            raise AssertionError(f"divisibility obstruction failed for N={particles}")

        permutation = list(reversed(range(dimension)))
        permuted_ranks = _cut_ranks(
            _permute_form(counterexample, permutation),
            dimension=dimension,
            particles=particles,
        )
        if permuted_ranks != counterexample_ranks:
            raise AssertionError(f"orbital permutation invariance failed for N={particles}")

        embedded_ranks = _cut_ranks(
            counterexample, dimension=dimension + 1, particles=particles
        )
        if embedded_ranks != counterexample_ranks:
            raise AssertionError(f"direct embedding invariance failed for N={particles}")

        perturbed = counterexample.copy()
        perturbation_support = tuple(range(1, particles + 1))
        perturbed[perturbation_support] = perturbed.get(perturbation_support, 0) + 1
        perturbed_ranks = _cut_ranks(
            perturbed, dimension=dimension, particles=particles
        )
        if perturbed_ranks[0] != dimension:
            raise AssertionError(f"full-support perturbation failed for N={particles}")

        first_flat = _flattening(
            first, dimension=dimension, particles=particles, cut=1
        )
        second_flat = _flattening(
            second, dimension=dimension, particles=particles, cut=1
        )
        first_rank = _rank(first_flat)
        second_rank = _rank(second_flat)
        sum_rank = _row_space_sum_rank(first_flat, second_flat)
        intersection_dimension = first_rank + second_rank - sum_rank
        channel_locking_deficit = sum_rank - counterexample_ranks[0]
        if (
            first_rank,
            second_rank,
            sum_rank,
            intersection_dimension,
            channel_locking_deficit,
        ) != (
            particles,
            particles,
            2 * particles,
            0,
            particles - 2,
        ):
            raise AssertionError(f"Slater-support overlap failed for N={particles}")

        cases.append(
            {
                "particles": particles,
                "one_particle_dimension": dimension,
                "slater_cut_ranks": slater_ranks,
                "expected_binomial_ranks": expected_slater_ranks,
                "counterexample_cut_ranks": counterexample_ranks,
                "permuted_cut_ranks": permuted_ranks,
                "embedded_cut_ranks": embedded_ranks,
                "perturbed_cut_ranks": perturbed_ranks,
                "one_cut_carrier_dimension": particles,
                "one_cut_counterexample_rank": counterexample_ranks[0],
                "rank_mod_carrier": counterexample_ranks[0] % particles,
                "component_image_intersection_dimension": intersection_dimension,
                "independent_component_image_sum_rank": sum_rank,
                "shared_domain_channel_locking_deficit": channel_locking_deficit,
                "slater_sum_length": 2,
            }
        )

    body: dict[str, object] = {
        "schema": "femps.statistics-carrier-obstruction.v1",
        "arithmetic": "exact rational",
        "counterexample": (
            "e1 wedge (e2 wedge e3 + e4 wedge e5) wedge "
            "e6 wedge ... wedge e_(N+2)"
        ),
        "verified_claim": (
            "Slater r1=N but counterexample r1=N+2, which is not divisible by N"
        ),
        "min_particles": min_particles,
        "max_particles": max_particles,
        "cases": cases,
    }
    body["certificate_sha256"] = _digest(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--min-particles", type=int, default=3)
    parser.add_argument("--max-particles", type=int, default=8)
    arguments = parser.parse_args()
    observed = build_certificate(arguments.min_particles, arguments.max_particles)
    if arguments.verify is not None:
        expected = json.loads(arguments.verify.read_text(encoding="utf-8"))
        if observed != expected:
            raise SystemExit("certificate mismatch")
        print(f"verified {arguments.verify} ({observed['certificate_sha256']})")
        return
    print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
