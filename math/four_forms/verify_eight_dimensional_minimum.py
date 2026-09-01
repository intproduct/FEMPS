"""Independently verify the eight-dimensional four-form certificate.

This standard-library-only verifier imports neither ``exact_contractions``
nor the FEMPS package.  It checks the recorded source transcription, reranks
all 94 nilpotent normal forms, and verifies the Cartan joint-eigenbasis and
finite-field hyperplane certificate.  Antonyan--Oeding Table 10 and the
theta-group orbit-closure theorem remain separate source-backed coverage
inputs; this script does not claim to re-prove them.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path


EXPECTED_PAYLOAD_SHA256 = (
    "44288f6097c7f56c746f3e3c39885fe707704acf47b957129e786afab044214b"
)
EXPECTED_SOURCE_SHA256 = (
    "bde922dcdf7766082b1fc2bb8d7f844ae24dff7aa0fe381504cb5cc68a453648"
)


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_keys(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} keys do not match the certificate format")
    return value


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


def _form_from_index_terms(
    raw_terms: object,
    *,
    dimension: int = 8,
    coefficients: bool = False,
) -> dict[tuple[int, ...], Fraction]:
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError("form terms must be a nonempty list")
    result: dict[tuple[int, ...], Fraction] = {}
    for raw_term in raw_terms:
        coefficient: object = 1
        indices_value: object = raw_term
        if coefficients:
            term = _require_keys(
                raw_term, {"coefficient", "indices"}, "coefficient-bearing term"
            )
            coefficient = term["coefficient"]
            indices_value = term["indices"]
        if not isinstance(coefficient, int):
            raise ValueError("term coefficients must be integers")
        if not isinstance(indices_value, list) or len(indices_value) != 4:
            raise ValueError("each four-form term must contain four indices")
        if not all(
            isinstance(index, int) and 1 <= index <= dimension
            for index in indices_value
        ):
            raise ValueError("form indices are outside the ambient space")
        zero_based = tuple(index - 1 for index in indices_value)
        sign = _sign(zero_based)
        if sign == 0:
            raise ValueError("form term contains a repeated index")
        key = tuple(sorted(zero_based))
        result[key] = result.get(key, Fraction(0)) + coefficient * sign
        if result[key] == 0:
            del result[key]
    if not result:
        raise ValueError("form terms cancel to zero")
    return result


def _matrix(
    form: dict[tuple[int, ...], Fraction], dimension: int, input_degree: int
) -> list[list[Fraction]]:
    columns = _basis(dimension, input_degree)
    rows = _basis(dimension, 4 - input_degree)
    return [
        [
            _sign(left + right)
            * form.get(tuple(sorted(left + right)), Fraction(0))
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


def _multiply(
    left: list[list[Fraction]], right: list[list[Fraction]]
) -> list[list[Fraction]]:
    return [
        [
            sum(
                (left[row][inner] * right[inner][column] for inner in range(28)),
                Fraction(0),
            )
            for column in range(28)
        ]
        for row in range(28)
    ]


def _matrix_vector(
    matrix: list[list[Fraction]], vector: list[Fraction]
) -> list[Fraction]:
    return [
        sum((entry * value for entry, value in zip(row, vector, strict=True)), Fraction(0))
        for row in matrix
    ]


def _verify_nilpotent_orbits(
    payload: dict[str, object], source: dict[str, object]
) -> dict[str, object]:
    raw_orbits = payload["nilpotent_orbits"]
    if not isinstance(raw_orbits, list) or len(raw_orbits) != 94:
        raise ValueError("expected exactly 94 nilpotent orbit representatives")
    if [row.get("orbit") for row in raw_orbits if isinstance(row, dict)] != list(
        range(1, 95)
    ):
        raise ValueError("nilpotent orbit numbering must be exactly 1,...,94")

    transcription: list[dict[str, object]] = []
    rank_vectors: list[tuple[int, ...]] = []
    concise_middle_ranks: list[int] = []
    minimizing_orbits: list[int] = []
    for raw_row in raw_orbits:
        row = _require_keys(
            raw_row,
            {
                "characteristic",
                "concise",
                "contraction_ranks",
                "normal_form_terms",
                "orbit",
                "orbit_dimension",
            },
            "nilpotent orbit row",
        )
        characteristic = row["characteristic"]
        if (
            not isinstance(characteristic, list)
            or len(characteristic) != 7
            or not all(isinstance(value, int) and value >= 0 for value in characteristic)
        ):
            raise ValueError(f"invalid characteristic for orbit {row['orbit']}")
        if not isinstance(row["orbit_dimension"], int):
            raise ValueError(f"invalid dimension for orbit {row['orbit']}")

        form = _form_from_index_terms(row["normal_form_terms"])
        ranks = tuple(_rank(_matrix(form, 8, degree)) for degree in range(5))
        if list(ranks) != row["contraction_ranks"]:
            raise ValueError(f"rank mismatch for nilpotent orbit {row['orbit']}")
        concise = ranks[1] == 8
        if row["concise"] is not concise:
            raise ValueError(f"conciseness mismatch for nilpotent orbit {row['orbit']}")
        rank_vectors.append(ranks)
        if concise:
            concise_middle_ranks.append(ranks[2])
        transcription.append(
            {
                key: row[key]
                for key in (
                    "orbit",
                    "characteristic",
                    "normal_form_terms",
                    "orbit_dimension",
                )
            }
        )

    if _digest(transcription) != source["source_transcription_sha256"]:
        raise ValueError("source transcription hash mismatch")
    if source["source_transcription_sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("source transcription differs from the reviewed artifact")

    minimum = min(concise_middle_ranks)
    for row, ranks in zip(raw_orbits, rank_vectors, strict=True):
        if ranks[1] == 8 and ranks[2] == minimum:
            minimizing_orbits.append(row["orbit"])

    summary = _require_keys(
        payload["nilpotent_summary"],
        {
            "concise_middle_rank_histogram",
            "concise_orbit_count",
            "minimizing_orbits",
            "minimum_concise_middle_rank",
            "orbit_count",
            "rank_vector_histogram",
        },
        "nilpotent summary",
    )
    expected_rank_histogram = [
        {"contraction_ranks": list(ranks), "count": count}
        for ranks, count in sorted(Counter(rank_vectors).items())
    ]
    expected_middle_histogram = {
        str(rank): count
        for rank, count in sorted(Counter(concise_middle_ranks).items())
    }
    expected_summary = {
        "orbit_count": 94,
        "concise_orbit_count": len(concise_middle_ranks),
        "minimum_concise_middle_rank": minimum,
        "minimizing_orbits": minimizing_orbits,
        "rank_vector_histogram": expected_rank_histogram,
        "concise_middle_rank_histogram": expected_middle_histogram,
    }
    if summary != expected_summary:
        raise ValueError("nilpotent summary does not match the reranked orbit table")
    if minimum != 12 or minimizing_orbits != [6]:
        raise ValueError("unexpected concise nilpotent minimum")
    return {
        "orbit_count": 94,
        "concise_orbit_count": len(concise_middle_ranks),
        "minimum_middle_rank": minimum,
        "minimizing_orbits": minimizing_orbits,
    }


def _verify_cartan(payload: dict[str, object]) -> dict[str, object]:
    certificate = _require_keys(
        payload["cartan_certificate"],
        {"basis", "finite_field_hyperplane_check", "joint_eigenpairs", "pair_basis"},
        "Cartan certificate",
    )
    expected_pair_basis = [
        "".join(str(index + 1) for index in pair) for pair in _basis(8, 2)
    ]
    if certificate["pair_basis"] != expected_pair_basis:
        raise ValueError("pair basis is missing, reordered, or malformed")

    raw_basis = certificate["basis"]
    if not isinstance(raw_basis, list) or len(raw_basis) != 7:
        raise ValueError("expected seven Cartan basis forms")
    matrices: list[list[list[Fraction]]] = []
    for raw_entry in raw_basis:
        entry = _require_keys(
            raw_entry, {"canonical_terms", "source_permutation"}, "Cartan basis row"
        )
        permutation = entry["source_permutation"]
        if (
            not isinstance(permutation, list)
            or sorted(permutation) != list(range(1, 9))
        ):
            raise ValueError("Cartan source row is not a permutation of 1,...,8")
        reconstructed = _form_from_index_terms(
            [permutation[:4], permutation[4:]]
        )
        recorded = _form_from_index_terms(entry["canonical_terms"], coefficients=True)
        if reconstructed != recorded:
            raise ValueError("Cartan canonical terms disagree with the source permutation")
        matrix = _matrix(reconstructed, 8, 2)
        if matrix != [list(row) for row in zip(*matrix, strict=True)]:
            raise ValueError("a Cartan middle contraction matrix is not symmetric")
        matrices.append(matrix)

    for left in range(7):
        for right in range(left):
            if _multiply(matrices[left], matrices[right]) != _multiply(
                matrices[right], matrices[left]
            ):
                raise ValueError("Cartan middle contraction matrices do not commute")

    raw_eigenpairs = certificate["joint_eigenpairs"]
    if not isinstance(raw_eigenpairs, list) or len(raw_eigenpairs) != 28:
        raise ValueError("expected 28 Cartan joint eigenpairs")
    weights: list[tuple[int, ...]] = []
    vectors: list[list[Fraction]] = []
    pair_positions = {pair: index for index, pair in enumerate(expected_pair_basis)}
    for raw_eigenpair in raw_eigenpairs:
        eigenpair = _require_keys(raw_eigenpair, {"vector", "weight"}, "joint eigenpair")
        weight = eigenpair["weight"]
        if (
            not isinstance(weight, list)
            or len(weight) != 7
            or not all(isinstance(value, int) for value in weight)
        ):
            raise ValueError("joint weight must be a seven-integer list")
        raw_vector = eigenpair["vector"]
        if not isinstance(raw_vector, list) or not raw_vector:
            raise ValueError("joint eigenvector must be a nonempty sparse list")
        vector = [Fraction(0) for _ in range(28)]
        seen_pairs: set[str] = set()
        for raw_component in raw_vector:
            component = _require_keys(
                raw_component, {"coefficient", "pair"}, "eigenvector component"
            )
            pair = component["pair"]
            coefficient = component["coefficient"]
            if pair not in pair_positions or pair in seen_pairs:
                raise ValueError("joint eigenvector contains an invalid or repeated pair")
            if not isinstance(coefficient, int) or coefficient == 0:
                raise ValueError("joint eigenvector coefficient must be a nonzero integer")
            vector[pair_positions[pair]] = coefficient
            seen_pairs.add(pair)
        for index, matrix in enumerate(matrices):
            if _matrix_vector(matrix, vector) != [weight[index] * value for value in vector]:
                raise ValueError("recorded vector is not a joint eigenvector")
        weights.append(tuple(weight))
        vectors.append(vector)

    if len(set(weights)) != 28 or _rank([list(column) for column in zip(*vectors)]) != 28:
        raise ValueError("recorded joint eigenvectors do not form a simple eigenbasis")

    finite_field = _require_keys(
        certificate["finite_field_hyperplane_check"],
        {
            "maximizing_nonzero_normals",
            "maximum_zero_weights",
            "nonzero_normals_checked",
            "prime",
            "projective_maximizer_count",
            "semisimple_nonzero_middle_rank_lower_bound",
        },
        "finite-field hyperplane check",
    )
    prime = finite_field["prime"]
    if prime != 3:
        raise ValueError("the reviewed hyperplane certificate uses F_3")
    zero_counts: list[int] = []
    for normal_number in range(1, prime**7):
        number = normal_number
        normal: list[int] = []
        for _ in range(7):
            normal.append(number % prime)
            number //= prime
        zero_counts.append(
            sum(
                sum(a * b for a, b in zip(normal, weight, strict=True)) % prime == 0
                for weight in weights
            )
        )
    maximum_zeros = max(zero_counts)
    maximizing_normals = zero_counts.count(maximum_zeros)
    expected_finite_field = {
        "prime": 3,
        "nonzero_normals_checked": 3**7 - 1,
        "maximum_zero_weights": maximum_zeros,
        "maximizing_nonzero_normals": maximizing_normals,
        "projective_maximizer_count": maximizing_normals // 2,
        "semisimple_nonzero_middle_rank_lower_bound": 28 - maximum_zeros,
    }
    if finite_field != expected_finite_field:
        raise ValueError("finite-field hyperplane statistics do not recompute")
    if maximum_zeros != 16 or 28 - maximum_zeros != 12:
        raise ValueError("unexpected semisimple middle-rank bound")
    return {
        "basis_dimension": 7,
        "joint_weight_count": 28,
        "maximum_weights_in_a_complex_hyperplane": maximum_zeros,
        "nonzero_semisimple_middle_rank_lower_bound": 28 - maximum_zeros,
    }


def _verify_conclusion(payload: dict[str, object]) -> dict[str, object]:
    conclusion = _require_keys(
        payload["conclusion"],
        {"fields", "quantity", "rational_witness_terms", "value"},
        "conclusion",
    )
    if (
        conclusion["quantity"] != "mu_4^K(8)"
        or conclusion["fields"] != ["C", "Qbar", "Q"]
        or conclusion["value"] != 12
    ):
        raise ValueError("unexpected eight-dimensional conclusion")
    witness = _form_from_index_terms(conclusion["rational_witness_terms"])
    ranks = tuple(_rank(_matrix(witness, 8, degree)) for degree in range(5))
    if ranks != (1, 8, 12, 8, 1):
        raise ValueError("rational witness does not have rank vector (1,8,12,8,1)")
    return {"value": 12, "rational_witness_rank_vector": list(ranks)}


def verify(path: Path) -> dict[str, object]:
    artifact = _require_keys(
        json.loads(path.read_text(encoding="utf-8")),
        {
            "ambient_dimension",
            "artifact_type",
            "artifact_version",
            "classification_field",
            "degree",
            "evidence_status",
            "indexing",
            "mathematical_payload",
            "orbit_relation",
            "payload_sha256",
            "source",
        },
        "artifact",
    )
    if (
        artifact["artifact_type"] != "femps.eight_dimensional_four_form_minimum"
        or artifact["artifact_version"] != 1
        or artifact["ambient_dimension"] != 8
        or artifact["degree"] != 4
        or artifact["classification_field"] != "C"
        or artifact["orbit_relation"] != "SL(8,C)"
    ):
        raise ValueError("artifact metadata does not describe the reviewed problem")
    source = _require_keys(
        artifact["source"],
        {
            "arxiv",
            "authors",
            "coverage",
            "doi",
            "source_transcription_sha256",
            "table",
            "title",
        },
        "classification source",
    )
    if (
        source["doi"] != "10.1090/mosc/332"
        or source["arxiv"] != "2205.09741v3"
        or source["table"] != 10
    ):
        raise ValueError("classification source is missing or changed")
    payload = _require_keys(
        artifact["mathematical_payload"],
        {"cartan_certificate", "conclusion", "nilpotent_orbits", "nilpotent_summary"},
        "mathematical payload",
    )
    if _digest(payload) != artifact["payload_sha256"]:
        raise ValueError("mathematical payload hash mismatch")
    if artifact["payload_sha256"] != EXPECTED_PAYLOAD_SHA256:
        raise ValueError("mathematical payload differs from the reviewed artifact")

    nilpotent = _verify_nilpotent_orbits(payload, source)
    cartan = _verify_cartan(payload)
    conclusion = _verify_conclusion(payload)
    return {
        "artifact": str(path),
        "cartan_certificate": cartan,
        "classification_exhaustiveness": (
            "source-backed by Antonyan--Oeding Table 10 and theta-group theory"
        ),
        "conclusion": conclusion,
        "nilpotent_certificate": nilpotent,
        "payload_sha256": artifact["payload_sha256"],
        "source_transcription_sha256": source["source_transcription_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.verify), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
