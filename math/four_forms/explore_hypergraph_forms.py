"""Seeded exploratory search for sparse concise four-forms.

Each coordinate four-form term contributes three complementary-pair edges to
the middle contraction matrix. This script screens random covering
4-uniform hypergraphs over F_2 with bit-exact elimination, then evaluates the
best candidate over Q using ``exact_contractions``. The search is explicitly
``numerical evidence``: a low rank modulo two need not lift to characteristic
zero, and random sampling proves no lower bound or chart coverage.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import random
from pathlib import Path


_EXACT_PATH = Path(__file__).with_name("exact_contractions.py")
_EXACT_SPEC = importlib.util.spec_from_file_location("four_form_exact_search", _EXACT_PATH)
if _EXACT_SPEC is None or _EXACT_SPEC.loader is None:
    raise ImportError(f"cannot load exact four-form utilities from {_EXACT_PATH}")
_exact = importlib.util.module_from_spec(_EXACT_SPEC)
_EXACT_SPEC.loader.exec_module(_exact)
canonical_form = _exact.canonical_form
four_form_hilbert_vector = _exact.four_form_hilbert_vector


def _gf2_rank(rows: list[int]) -> int:
    pivots: dict[int, int] = {}
    for raw_row in rows:
        row = raw_row
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    return len(pivots)


def _covering_hypergraph(
    rng: random.Random, ambient_dimension: int, term_count: int
) -> tuple[tuple[int, int, int, int], ...]:
    if 4 * term_count < ambient_dimension:
        raise ValueError("term_count cannot cover the ambient dimension")
    while True:
        incidences = list(range(ambient_dimension))
        incidences.extend(
            rng.randrange(ambient_dimension)
            for _ in range(4 * term_count - ambient_dimension)
        )
        rng.shuffle(incidences)
        terms = tuple(
            sorted(
                tuple(sorted(incidences[offset : offset + 4]))
                for offset in range(0, len(incidences), 4)
            )
        )
        if all(len(set(term)) == 4 for term in terms) and len(set(terms)) == term_count:
            return tuple(sorted(terms))


def _gf2_contraction_ranks(
    terms: tuple[tuple[int, int, int, int], ...], ambient_dimension: int
) -> tuple[int, int]:
    triples = sorted({triple for term in terms for triple in itertools.combinations(term, 3)})
    triple_index = {triple: index for index, triple in enumerate(triples)}
    first_rows = [0] * len(triples)
    for term in terms:
        for omitted in term:
            triple = tuple(index for index in term if index != omitted)
            first_rows[triple_index[triple]] ^= 1 << omitted
    first_rank = _gf2_rank(first_rows)

    pairs = sorted({pair for term in terms for pair in itertools.combinations(term, 2)})
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    middle_rows = [0] * len(pairs)
    for term in terms:
        for left in itertools.combinations(term, 2):
            right = tuple(index for index in term if index not in left)
            left_index = pair_index[tuple(sorted(left))]
            right_index = pair_index[tuple(sorted(right))]
            middle_rows[left_index] ^= 1 << right_index
    return first_rank, _gf2_rank(middle_rows)


def search(
    *,
    ambient_dimension: int,
    term_count: int,
    samples: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    best_terms: tuple[tuple[int, int, int, int], ...] | None = None
    best_middle_rank: int | None = None
    concise_samples = 0
    rank_histogram: dict[int, int] = {}
    for _ in range(samples):
        terms = _covering_hypergraph(rng, ambient_dimension, term_count)
        first_rank, middle_rank = _gf2_contraction_ranks(terms, ambient_dimension)
        if first_rank != ambient_dimension:
            continue
        concise_samples += 1
        rank_histogram[middle_rank] = rank_histogram.get(middle_rank, 0) + 1
        if best_middle_rank is None or middle_rank < best_middle_rank:
            best_middle_rank = middle_rank
            best_terms = terms

    if best_terms is None:
        return {
            "evidence_status": "numerical evidence",
            "ambient_dimension": ambient_dimension,
            "term_count": term_count,
            "samples": samples,
            "seed": seed,
            "screen_field": "F_2",
            "concise_samples": 0,
            "best_candidate": None,
        }

    rational_form = canonical_form({term: 1 for term in best_terms})
    rational_ranks = four_form_hilbert_vector(rational_form, ambient_dimension)
    return {
        "evidence_status": "numerical evidence",
        "ambient_dimension": ambient_dimension,
        "term_count": term_count,
        "samples": samples,
        "seed": seed,
        "screen_field": "F_2",
        "concise_samples": concise_samples,
        "middle_rank_histogram": dict(sorted(rank_histogram.items())),
        "best_candidate": {
            "terms": [list(term) for term in best_terms],
            "f2_middle_rank": best_middle_rank,
            "q_contraction_ranks": list(rational_ranks),
        },
        "warning": "Random finite-field screening proves neither sharpness nor characteristic-zero lifting.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ambient-dimension", type=int, default=16)
    parser.add_argument("--terms", type=int, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=270016)
    arguments = parser.parse_args()
    result = search(
        ambient_dimension=arguments.ambient_dimension,
        term_count=arguments.terms,
        samples=arguments.samples,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
