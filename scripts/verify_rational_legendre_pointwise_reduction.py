"""Exact checks for the shifted rational-Legendre pointwise reduction.

This verifies the interpolation and exterior/Cayley identities only. It does
not reimplement or certify the CHSS SAT gadgets.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from math import comb
import random


def legendre_coefficients(count: int) -> list[list[Fraction]]:
    """Return coefficients of ell_r(t)=P_r(2t-1), with P_r(1)=1."""

    return [
        [
            Fraction((-1) ** (degree - power) * comb(degree, power) * comb(degree + power, power))
            for power in range(degree + 1)
        ]
        for degree in range(count)
    ]


def evaluate(poly: list[Fraction], point: Fraction) -> Fraction:
    value = Fraction(0)
    for coefficient in reversed(poly):
        value = value * point + coefficient
    return value


def inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    size = len(matrix)
    work = [row[:] + [Fraction(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = next(row for row in range(column, size) if work[row][column])
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(size):
            if row == column:
                continue
            scale = work[row][column]
            work[row] = [a - scale * b for a, b in zip(work[row], work[column], strict=True)]
    return [row[size:] for row in work]


Scalar = int | Fraction
Matrix = tuple[tuple[Scalar, Scalar], tuple[Scalar, Scalar]]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    return (
        (a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]),
    )


def matadd(a: Matrix, b: Matrix, sign: int = 1) -> Matrix:
    return tuple(tuple(a[i][j] + sign * b[i][j] for j in range(2)) for i in range(2))  # type: ignore[return-value]


def parity(order: tuple[int, ...]) -> int:
    inversions = sum(order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))
    return -1 if inversions % 2 else 1


def cdet(entries: list[list[Matrix]]) -> Matrix:
    size = len(entries)
    result: Matrix = ((0, 0), (0, 0))
    identity: Matrix = ((1, 0), (0, 1))
    for order in permutations(range(size)):
        product = identity
        for row, column in enumerate(order):
            product = matmul(product, entries[row][column])
        result = matadd(result, product, parity(order))
    return result


def boundary(matrix: Matrix) -> Scalar:
    # u=e1 and v=e1+e2
    return matrix[0][0] + matrix[0][1]


def verify(size: int, seed: int) -> None:
    polys = legendre_coefficients(size)
    nodes = [Fraction(j, size + 1) for j in range(1, size + 1)]
    evaluation = [[evaluate(polys[r], node) for r in range(size)] for node in nodes]
    evaluation_inverse = inverse(evaluation)
    for k in range(size):
        for j in range(size):
            value = sum(evaluation[k][r] * evaluation_inverse[r][j] for r in range(size))
            assert value == Fraction(k == j)

    determinant = Fraction(1)
    leading_product = 1
    vandermonde = Fraction(1)
    for degree in range(size):
        leading_product *= comb(2 * degree, degree)
    for left in range(size):
        for right in range(left + 1, size):
            vandermonde *= nodes[right] - nodes[left]
    determinant = Fraction(leading_product) * vandermonde
    assert determinant != 0

    generator = random.Random(seed)
    entries: list[list[Matrix]] = []
    for _ in range(size):
        row = []
        for _ in range(size):
            row.append(tuple(tuple(generator.randint(-2, 2) for _ in range(2)) for _ in range(2)))  # type: ignore[arg-type]
        entries.append(row)

    direct = boundary(cdet(entries))
    alternating: Scalar = 0
    for order in permutations(range(size)):
        product: Matrix = ((1, 0), (0, 1))
        for site, node_index in enumerate(order):
            functional_value: Matrix = ((0, 0), (0, 0))
            for basis_index in range(size):
                scaled = tuple(
                    tuple(evaluation[node_index][basis_index] * entries[site][basis_index][row][column] for column in range(2))
                    for row in range(2)
                )
                functional_value = matadd(functional_value, scaled)  # type: ignore[arg-type]
            product = matmul(product, functional_value)
        alternating += parity(order) * boundary(product)
    assert alternating == determinant * direct

    max_bits = max(
        max(value.numerator.bit_length(), value.denominator.bit_length())
        for row in evaluation_inverse
        for value in row
    )
    print(
        f"n={size}: shifted-Legendre determinant/inverse and Cayley point formula pass; "
        f"max inverse bits={max_bits}"
    )


def main() -> None:
    for size in range(2, 7):
        verify(size, 4100 + size)
    print("rational Legendre pointwise reduction checks passed")


if __name__ == "__main__":
    main()
