"""Exact small-order checks for the rational-Legendre pointwise reduction.

This verifies the interpolation and exterior/Cayley identities only. It does
not reimplement or certify the CHSS SAT gadgets.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from math import factorial
import random


def poly_add(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0) for _ in range(max(len(a), len(b)))]
    for i, value in enumerate(a):
        out[i] += value
    for i, value in enumerate(b):
        out[i] += value
    return out


def poly_scale(a: list[Fraction], scale: Fraction) -> list[Fraction]:
    return [scale * value for value in a]


def legendre_coefficients(count: int) -> list[list[Fraction]]:
    values = [[Fraction(1)]]
    if count == 1:
        return values
    values.append([Fraction(0), Fraction(1)])
    for degree in range(1, count - 1):
        x_times = [Fraction(0)] + values[degree]
        first = poly_scale(x_times, Fraction(2 * degree + 1, degree + 1))
        second = poly_scale(values[degree - 1], Fraction(-degree, degree + 1))
        values.append(poly_add(first, second))
    return values


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


Matrix = tuple[tuple[int, int], tuple[int, int]]


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


def boundary(matrix: Matrix) -> int:
    # u=e1 and v=e1+e2
    return matrix[0][0] + matrix[0][1]


def verify(size: int, seed: int) -> None:
    polys = legendre_coefficients(size)
    nodes = [Fraction(-1) + Fraction(2 * j, size + 1) for j in range(1, size + 1)]
    evaluation = [[evaluate(polys[r], node) for r in range(size)] for node in nodes]
    evaluation_inverse = inverse(evaluation)
    for k in range(size):
        for j in range(size):
            value = sum(evaluation[k][r] * evaluation_inverse[r][j] for r in range(size))
            assert value == Fraction(k == j)

    generator = random.Random(seed)
    entries: list[list[Matrix]] = []
    for _ in range(size):
        row = []
        for _ in range(size):
            row.append(tuple(tuple(generator.randint(-2, 2) for _ in range(2)) for _ in range(2)))  # type: ignore[arg-type]
        entries.append(row)

    direct = boundary(cdet(entries))
    alternating = 0
    for order in permutations(range(size)):
        product: Matrix = ((1, 0), (0, 1))
        for site, node_index in enumerate(order):
            # L_j(xi_k)=delta_jk, so F_site(xi_node_index)=H_site,node_index.
            product = matmul(product, entries[site][node_index])
        alternating += parity(order) * boundary(product)
    assert alternating == direct

    max_bits = max(
        max(value.numerator.bit_length(), value.denominator.bit_length())
        for row in evaluation_inverse
        for value in row
    )
    print(f"n={size}: exact interpolation and Cayley point value pass; max inverse bits={max_bits}")


def main() -> None:
    for size in range(2, 7):
        verify(size, 4100 + size)
    print("rational Legendre pointwise reduction checks passed")


if __name__ == "__main__":
    main()

