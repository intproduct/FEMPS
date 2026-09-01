"""Exact contraction matrices and ranks for alternating forms.

This module is intentionally standard-library-only. It is research support
code, not part of the production FEMPS package and not an independent
certificate verifier.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from itertools import combinations


IndexTuple = tuple[int, ...]
Scalar = int | Fraction
Form = dict[IndexTuple, Fraction]
Matrix = list[list[Fraction]]


def _as_fraction(value: Scalar) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("exact coefficients must be int or Fraction")
    return Fraction(value)


def permutation_sign(indices: Sequence[int]) -> int:
    """Return the sign needed to sort distinct ``indices`` increasingly."""

    if len(set(indices)) != len(indices):
        return 0
    inversions = sum(
        indices[left] > indices[right]
        for left in range(len(indices))
        for right in range(left + 1, len(indices))
    )
    return -1 if inversions % 2 else 1


def canonical_form(
    terms: Mapping[Sequence[int], Scalar] | Iterable[tuple[Sequence[int], Scalar]],
    *,
    degree: int | None = None,
) -> Form:
    """Combine signed terms into increasing-index exact exterior coordinates."""

    items = terms.items() if isinstance(terms, Mapping) else terms
    result: Form = {}
    inferred_degree = degree
    for raw_indices, raw_coefficient in items:
        indices = tuple(raw_indices)
        if inferred_degree is None:
            inferred_degree = len(indices)
        if len(indices) != inferred_degree:
            raise ValueError("all exterior terms must have the same degree")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
            raise TypeError("basis indices must be integers")
        if any(index < 0 for index in indices):
            raise ValueError("basis indices must be nonnegative")
        sign = permutation_sign(indices)
        if sign == 0:
            continue
        key = tuple(sorted(indices))
        coefficient = sign * _as_fraction(raw_coefficient)
        result[key] = result.get(key, Fraction(0)) + coefficient
        if result[key] == 0:
            del result[key]
    if inferred_degree is None:
        if degree is None:
            raise ValueError("degree is required for an empty form")
        inferred_degree = degree
    if inferred_degree < 0:
        raise ValueError("degree must be nonnegative")
    return result


def form_degree(form: Mapping[IndexTuple, Scalar], *, empty_degree: int | None = None) -> int:
    """Return the common exterior degree, requiring a hint for the zero form."""

    if not form:
        if empty_degree is None:
            raise ValueError("the zero form has no inferred degree")
        return empty_degree
    degrees = {len(indices) for indices in form}
    if len(degrees) != 1:
        raise ValueError("form keys have inconsistent degrees")
    return degrees.pop()


def exterior_basis(ambient_dimension: int, degree: int) -> tuple[IndexTuple, ...]:
    """Return the lexicographically ordered increasing exterior basis."""

    if ambient_dimension < 0:
        raise ValueError("ambient_dimension must be nonnegative")
    if not 0 <= degree <= ambient_dimension:
        raise ValueError("degree must lie between zero and ambient_dimension")
    return tuple(combinations(range(ambient_dimension), degree))


def _validate_ambient(form: Mapping[IndexTuple, Scalar], ambient_dimension: int) -> int:
    degree = form_degree(form)
    canonical = canonical_form(form, degree=degree)
    if canonical != {key: _as_fraction(value) for key, value in form.items()}:
        raise ValueError("form keys must be increasing, unique exterior coordinates")
    if any(index >= ambient_dimension for key in canonical for index in key):
        raise ValueError("form index lies outside the ambient dimension")
    return degree


def contraction_matrix(
    form: Mapping[IndexTuple, Scalar],
    ambient_dimension: int,
    input_degree: int,
) -> Matrix:
    """Build ``C_j(form)`` with entries ``form(e_I wedge e_J)`` exactly."""

    degree = _validate_ambient(form, ambient_dimension)
    if not 0 <= input_degree <= degree:
        raise ValueError("input_degree must lie between zero and the form degree")
    normalized = {key: _as_fraction(value) for key, value in form.items()}
    columns = exterior_basis(ambient_dimension, input_degree)
    rows = exterior_basis(ambient_dimension, degree - input_degree)
    matrix: Matrix = []
    for right in rows:
        row: list[Fraction] = []
        for left in columns:
            joined = left + right
            sign = permutation_sign(joined)
            coefficient = normalized.get(tuple(sorted(joined)), Fraction(0))
            row.append(sign * coefficient)
        matrix.append(row)
    return matrix


def rational_rank(matrix: Sequence[Sequence[Scalar]]) -> int:
    """Compute matrix rank over ``Q`` by exact row reduction."""

    if not matrix:
        return 0
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    work = [[_as_fraction(value) for value in row] for row in matrix]
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (row for row in range(pivot_row, len(work)) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        pivot_value = work[pivot_row][column]
        work[pivot_row] = [value / pivot_value for value in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or work[row][column] == 0:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(work[row], work[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def rank_mod_prime(matrix: Sequence[Sequence[int]], prime: int) -> int:
    """Compute exact rank over the recorded prime field ``F_prime``."""

    if prime < 2 or any(prime % divisor == 0 for divisor in range(2, int(prime**0.5) + 1)):
        raise ValueError("prime must be prime")
    if not matrix:
        return 0
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    work: list[list[int]] = []
    for row in matrix:
        converted: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("finite-field matrices must have integer entries")
            converted.append(value % prime)
        work.append(converted)
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (row for row in range(pivot_row, len(work)) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        inverse = pow(work[pivot_row][column], -1, prime)
        work[pivot_row] = [(value * inverse) % prime for value in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row or work[row][column] == 0:
                continue
            factor = work[row][column]
            work[row] = [
                (value - factor * pivot_entry) % prime
                for value, pivot_entry in zip(work[row], work[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def contraction_rank(
    form: Mapping[IndexTuple, Scalar],
    ambient_dimension: int,
    input_degree: int,
) -> int:
    """Return the exact rational rank of a contraction map."""

    return rational_rank(contraction_matrix(form, ambient_dimension, input_degree))


def four_form_hilbert_vector(
    form: Mapping[IndexTuple, Scalar], ambient_dimension: int
) -> tuple[int, int, int, int, int]:
    """Return ``(rank C_0, ..., rank C_4)`` for a nonzero four-form."""

    if form_degree(form) != 4:
        raise ValueError("expected a four-form")
    ranks = tuple(
        contraction_rank(form, ambient_dimension, degree) for degree in range(5)
    )
    return ranks  # type: ignore[return-value]


def is_concise(form: Mapping[IndexTuple, Scalar], ambient_dimension: int) -> bool:
    """Return whether the one-vector contraction has full ambient rank."""

    return contraction_rank(form, ambient_dimension, 1) == ambient_dimension


def volume_form(indices: Sequence[int], coefficient: Scalar = 1) -> Form:
    """Return one decomposable coordinate volume form."""

    return canonical_form([(indices, coefficient)])


def direct_sum(*forms: tuple[Mapping[IndexTuple, Scalar], int]) -> tuple[Form, int]:
    """Place forms in disjoint consecutive ambient blocks and add them."""

    if not forms:
        raise ValueError("at least one form is required")
    terms: list[tuple[IndexTuple, Scalar]] = []
    offset = 0
    degree: int | None = None
    for form, dimension in forms:
        current_degree = _validate_ambient(form, dimension)
        if degree is None:
            degree = current_degree
        elif current_degree != degree:
            raise ValueError("direct-sum forms must have equal degree")
        terms.extend((tuple(index + offset for index in key), value) for key, value in form.items())
        offset += dimension
    return canonical_form(terms, degree=degree), offset


def permute_basis(
    form: Mapping[IndexTuple, Scalar], permutation: Sequence[int]
) -> Form:
    """Relabel ``e_i`` as ``e_permutation[i]`` and recanonicalize the form."""

    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("permutation must contain each ambient index exactly once")
    _validate_ambient(form, len(permutation))
    return canonical_form(
        [(tuple(permutation[index] for index in key), value) for key, value in form.items()]
    )


def hodge_dual(form: Mapping[IndexTuple, Scalar], ambient_dimension: int) -> Form:
    """Apply the coordinate Hodge star for the oriented orthonormal basis."""

    degree = _validate_ambient(form, ambient_dimension)
    all_indices = set(range(ambient_dimension))
    terms: list[tuple[IndexTuple, Scalar]] = []
    for key, value in form.items():
        complement = tuple(sorted(all_indices.difference(key)))
        terms.append((complement, permutation_sign(key + complement) * _as_fraction(value)))
    return canonical_form(terms, degree=ambient_dimension - degree)
