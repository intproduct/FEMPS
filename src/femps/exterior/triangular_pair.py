"""Exact LC-AGP collapse for 2 x 2 upper-triangular pair coefficients."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import math

import torch


def _fraction_solve(
    matrix: list[list[Fraction]], right_hand_side: list[Fraction]
) -> list[Fraction]:
    order = len(matrix)
    if order == 0 or any(len(row) != order for row in matrix):
        raise ValueError("matrix must be nonempty and square")
    if len(right_hand_side) != order:
        raise ValueError("right-hand side has incompatible length")
    augmented = [
        [*row, right_hand_side[index]] for index, row in enumerate(matrix)
    ]
    for column in range(order):
        pivot = next(
            (
                row
                for row in range(column, order)
                if augmented[row][column]
            ),
            None,
        )
        if pivot is None:
            raise RuntimeError("triangular interpolation grid is singular")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        pivot_value = augmented[column][column]
        augmented[column] = [
            value / pivot_value for value in augmented[column]
        ]
        for row in range(order):
            if row == column or not augmented[row][column]:
                continue
            scale = augmented[row][column]
            augmented[row] = [
                value - scale * pivot_entry
                for value, pivot_entry in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[index][-1] for index in range(order)]


@lru_cache(maxsize=None)
def _offdiagonal_power_decomposition(
    pairs: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[Fraction, ...]]:
    """Return rational weights for ``sum F^t G H^(M-1-t)``.

    The powers ``(F + b*G + c*H)^M`` are sampled on the triangular integer
    grid ``b,c >= 0`` and ``b+c <= M``.  Multivariate Newton interpolation
    makes this an exact basis of the degree-M homogeneous polynomials.
    """

    if pairs < 1:
        raise ValueError("pairs must be positive")
    exponent_pairs = tuple(
        (g_exponent, h_exponent)
        for total in range(pairs + 1)
        for g_exponent in range(total + 1)
        for h_exponent in [total - g_exponent]
    )
    grid = exponent_pairs
    matrix = []
    target = []
    factorial = math.factorial
    for g_exponent, h_exponent in exponent_pairs:
        f_exponent = pairs - g_exponent - h_exponent
        multinomial = Fraction(
            factorial(pairs),
            factorial(f_exponent)
            * factorial(g_exponent)
            * factorial(h_exponent),
        )
        matrix.append(
            [
                multinomial
                * Fraction(g_scale**g_exponent)
                * Fraction(h_scale**h_exponent)
                for g_scale, h_scale in grid
            ]
        )
        target.append(Fraction(1 if g_exponent == 1 else 0))
    weights = _fraction_solve(matrix, target)
    return grid, tuple(weights)


def triangular_pair_lc_agp_decomposition(
    pair_matrices: torch.Tensor,
    pairs: int,
    left_boundary: torch.Tensor,
    right_boundary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse a 2 x 2 upper-triangular matrix-pair power to LC-AGP.

    ``pair_matrices[i,j]`` must lie in the upper-triangular algebra ``T_2``.
    If its three scalar pair forms are ``F,G,H``, then the off-diagonal block
    of its M-th power is ``sum_t F^t G H^(M-1-t)``.  Exact rational
    interpolation writes this polynomial as at most ``binom(M+2,2)`` scalar
    M-th powers.  Two additional terms cover arbitrary diagonal boundary
    contributions.

    The returned pair matrices and amplitudes define an exact finite LC-AGP
    representation in exact arithmetic.  Floating execution is intended as a
    small-system theorem oracle, not as a conditioned production solver.
    Derivative equivalence is on the physical-skew, virtual-upper-triangular
    parameter submanifold; no equality is asserted for forbidden lower-block
    ambient directions.
    """

    if pair_matrices.ndim != 4 or pair_matrices.shape[2:] != (2, 2):
        raise ValueError("pair_matrices must have shape (D, D, 2, 2)")
    dimension, second_dimension = pair_matrices.shape[:2]
    if dimension != second_dimension or 2 * pairs > dimension or pairs < 1:
        raise ValueError("require square physical indices and 1 <= 2*pairs <= D")
    if left_boundary.shape != (2,) or right_boundary.shape != (2,):
        raise ValueError("boundaries must have shape (2,)")
    if (
        left_boundary.dtype != pair_matrices.dtype
        or right_boundary.dtype != pair_matrices.dtype
        or left_boundary.device != pair_matrices.device
        or right_boundary.device != pair_matrices.device
    ):
        raise ValueError("pair matrices and boundaries must share dtype/device")
    if torch.count_nonzero(pair_matrices[:, :, 1, 0]).item():
        raise ValueError("coefficient matrices must be upper triangular")
    skew_residual = torch.max(
        torch.abs(pair_matrices + pair_matrices.transpose(0, 1))
    )
    tolerance = (
        dimension
        * torch.finfo(pair_matrices.real.dtype).eps
        * max(float(torch.linalg.vector_norm(pair_matrices).detach()), 1.0)
    )
    if float(skew_residual.detach()) > tolerance:
        raise ValueError("pair matrices must be skew in the physical indices")

    left_diagonal = pair_matrices[:, :, 0, 0]
    radical = pair_matrices[:, :, 0, 1]
    right_diagonal = pair_matrices[:, :, 1, 1]
    grid, rational_weights = _offdiagonal_power_decomposition(pairs)
    forms = []
    amplitudes = []
    offdiagonal_boundary = left_boundary[0] * right_boundary[1]
    for (g_scale, h_scale), weight in zip(
        grid, rational_weights, strict=True
    ):
        if not weight:
            continue
        forms.append(
            left_diagonal + g_scale * radical + h_scale * right_diagonal
        )
        amplitudes.append(
            offdiagonal_boundary
            * pair_matrices.new_tensor(weight.numerator / weight.denominator)
        )
    forms.extend([left_diagonal, right_diagonal])
    amplitudes.extend(
        [
            left_boundary[0] * right_boundary[0],
            left_boundary[1] * right_boundary[1],
        ]
    )
    return torch.stack(forms), torch.stack(amplitudes)


def triangular_pair_lc_agp_term_bound(pairs: int) -> int:
    """Return the proved LC-AGP term bound for arbitrary T2 boundaries."""

    if pairs < 1:
        raise ValueError("pairs must be positive")
    return math.comb(pairs + 2, 2) + 2
