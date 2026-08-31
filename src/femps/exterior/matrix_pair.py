"""Small exact oracles for virtual-matrix-valued pair powers.

The ansatz in this module replaces the scalar coefficients of an AGP two-form
by matrices acting on a finite virtual space.  It is a Phase 13 research
object, not yet a scalable contraction algorithm: exterior supports are
enumerated explicitly.
"""

from __future__ import annotations

import itertools
import math

import torch

from .reference import (
    exterior_coefficients_to_tensor,
    one_body_expectation_exterior_coefficients,
)


def _validate_matrix_pair_data(
    pair_matrices: torch.Tensor,
    pairs: int,
    left_boundary: torch.Tensor,
    right_boundary: torch.Tensor,
) -> tuple[int, int]:
    if pair_matrices.ndim != 4:
        raise ValueError("pair_matrices must have shape (D, D, chi, chi)")
    dimension, second_dimension, left_bond, right_bond = pair_matrices.shape
    if dimension != second_dimension or left_bond != right_bond:
        raise ValueError("pair_matrices must have shape (D, D, chi, chi)")
    if pairs < 0 or 2 * pairs > dimension:
        raise ValueError("pairs must satisfy 0 <= 2*pairs <= D")
    if left_boundary.shape != (left_bond,) or right_boundary.shape != (left_bond,):
        raise ValueError("boundaries must both have shape (chi,)")
    if (
        left_boundary.dtype != pair_matrices.dtype
        or right_boundary.dtype != pair_matrices.dtype
        or left_boundary.device != pair_matrices.device
        or right_boundary.device != pair_matrices.device
    ):
        raise ValueError("pair matrices and boundaries must share dtype and device")
    return dimension, left_bond


def matrix_pair_skew_residual(pair_matrices: torch.Tensor) -> torch.Tensor:
    """Return the largest absolute violation of physical-index skew symmetry."""

    if pair_matrices.ndim != 4:
        raise ValueError("pair_matrices must have shape (D, D, chi, chi)")
    return torch.max(torch.abs(pair_matrices + pair_matrices.transpose(0, 1)))


def matrix_pair_exterior_matrices(
    pair_matrices: torch.Tensor,
    pairs: int,
) -> torch.Tensor:
    """Return matrix coefficients of ``Omega^pairs / pairs!``.

    ``Omega = sum_{i<j} B[i,j] e_i wedge e_j`` has matrix coefficients
    ``B[i,j]``.  The output is ordered lexicographically by increasing
    ``2*pairs``-element supports and has shape ``(binom(D,2*pairs),chi,chi)``.
    The recurrence preserves multiplication order, so it applies when the
    coefficient matrices do not commute.
    """

    if pair_matrices.ndim != 4:
        raise ValueError("pair_matrices must have shape (D, D, chi, chi)")
    dimension, second_dimension, left_bond, right_bond = pair_matrices.shape
    if dimension != second_dimension or left_bond != right_bond:
        raise ValueError("pair_matrices must have shape (D, D, chi, chi)")
    if pairs < 0 or 2 * pairs > dimension:
        raise ValueError("pairs must satisfy 0 <= 2*pairs <= D")

    coefficients: dict[tuple[int, ...], torch.Tensor] = {
        (): torch.eye(left_bond, dtype=pair_matrices.dtype, device=pair_matrices.device)
    }
    for degree in range(1, pairs + 1):
        next_coefficients: dict[tuple[int, ...], torch.Tensor] = {}
        for support in itertools.combinations(range(dimension), 2 * degree):
            terms = []
            for first_position, second_position in itertools.combinations(
                range(2 * degree), 2
            ):
                pair = (support[first_position], support[second_position])
                previous_support = tuple(
                    index
                    for position, index in enumerate(support)
                    if position not in (first_position, second_position)
                )
                sign = -1 if (first_position + second_position + 1) % 2 else 1
                terms.append(
                    sign
                    * (
                        coefficients[previous_support]
                        @ pair_matrices[pair[0], pair[1]]
                    )
                )
            next_coefficients[support] = torch.stack(terms).sum(dim=0) / degree
        coefficients = next_coefficients
    return torch.stack(list(coefficients.values()))


def matrix_pair_exterior_coefficients(
    pair_matrices: torch.Tensor,
    pairs: int,
    left_boundary: torch.Tensor,
    right_boundary: torch.Tensor,
) -> torch.Tensor:
    """Contract virtual boundaries around matrix-valued pair-power coefficients."""

    _validate_matrix_pair_data(pair_matrices, pairs, left_boundary, right_boundary)
    matrices = matrix_pair_exterior_matrices(pair_matrices, pairs)
    return torch.einsum("a,sab,b->s", left_boundary, matrices, right_boundary)


def matrix_pair_tensor(
    pair_matrices: torch.Tensor,
    pairs: int,
    left_boundary: torch.Tensor,
    right_boundary: torch.Tensor,
) -> torch.Tensor:
    """Materialize the normalized-convention particle tensor."""

    dimension, _ = _validate_matrix_pair_data(
        pair_matrices, pairs, left_boundary, right_boundary
    )
    coefficients = matrix_pair_exterior_coefficients(
        pair_matrices, pairs, left_boundary, right_boundary
    )
    return exterior_coefficients_to_tensor(coefficients, dimension, 2 * pairs)


def matrix_pair_femps_cores(
    pair_matrices: torch.Tensor,
    pairs: int,
    left_boundary: torch.Tensor,
    right_boundary: torch.Tensor,
) -> list[torch.Tensor]:
    """Embed a matrix-valued pair power into one-form matrix-wedge cores.

    Two particle sites emit each pair form. Bonds alternate between ``chi``
    and ``chi*D`` (with open boundary bonds equal to one), so the embedding is
    polynomial and does not enumerate exterior supports.
    """

    dimension, bond = _validate_matrix_pair_data(
        pair_matrices, pairs, left_boundary, right_boundary
    )
    if pairs < 1:
        raise ValueError("the open-chain FEMPS embedding requires pairs >= 1")
    cores: list[torch.Tensor] = []
    for pair_index in range(pairs):
        left_bond = 1 if pair_index == 0 else bond
        odd = torch.zeros(
            left_bond,
            dimension,
            bond * dimension,
            dtype=pair_matrices.dtype,
            device=pair_matrices.device,
        )
        for virtual in range(bond):
            for physical in range(dimension):
                value = (
                    left_boundary[virtual] / math.factorial(pairs)
                    if pair_index == 0
                    else torch.ones(
                        (), dtype=pair_matrices.dtype, device=pair_matrices.device
                    )
                )
                odd[0 if pair_index == 0 else virtual, physical, virtual * dimension + physical] = value
        cores.append(odd)

        right_bond = 1 if pair_index == pairs - 1 else bond
        even = torch.zeros(
            bond * dimension,
            dimension,
            right_bond,
            dtype=pair_matrices.dtype,
            device=pair_matrices.device,
        )
        for left_virtual in range(bond):
            for first_physical in range(dimension):
                for second_physical in range(first_physical + 1, dimension):
                    row = pair_matrices[first_physical, second_physical, left_virtual]
                    if pair_index == pairs - 1:
                        even[
                            left_virtual * dimension + first_physical,
                            second_physical,
                            0,
                        ] = row @ right_boundary
                    else:
                        even[
                            left_virtual * dimension + first_physical,
                            second_physical,
                            :,
                        ] = row
        cores.append(even)
    return cores


def matrix_pair_norm(
    pair_matrices: torch.Tensor,
    pairs: int,
    left_boundary: torch.Tensor,
    right_boundary: torch.Tensor,
) -> torch.Tensor:
    """Return the exact norm via explicit increasing-basis coefficients."""

    coefficients = matrix_pair_exterior_coefficients(
        pair_matrices, pairs, left_boundary, right_boundary
    )
    return torch.vdot(coefficients, coefficients).real


def matrix_pair_one_body_expectation(
    pair_matrices: torch.Tensor,
    pairs: int,
    left_boundary: torch.Tensor,
    right_boundary: torch.Tensor,
    operator: torch.Tensor,
) -> torch.Tensor:
    """Return an unnormalized one-body matrix element in the exterior basis."""

    coefficients = matrix_pair_exterior_coefficients(
        pair_matrices, pairs, left_boundary, right_boundary
    )
    return one_body_expectation_exterior_coefficients(coefficients, operator, 2 * pairs)


def matrix_pair_n4_anticommutator(
    pair_matrices: torch.Tensor,
) -> torch.Tensor:
    """Return the explicit N=4 symmetrized noncommutative Pfaffian.

    For each ``i<j<k<l`` the matrix coefficient is

    ``({B_ij,B_kl} - {B_ik,B_jl} + {B_il,B_jk}) / 2``.
    """

    if pair_matrices.ndim != 4:
        raise ValueError("pair_matrices must have shape (D, D, chi, chi)")
    dimension, second_dimension, left_bond, right_bond = pair_matrices.shape
    if dimension != second_dimension or left_bond != right_bond:
        raise ValueError("pair_matrices must have shape (D, D, chi, chi)")

    def anticommutator(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return first @ second + second @ first

    values = []
    for first, second, third, fourth in itertools.combinations(range(dimension), 4):
        values.append(
            (
                anticommutator(
                    pair_matrices[first, second], pair_matrices[third, fourth]
                )
                - anticommutator(
                    pair_matrices[first, third], pair_matrices[second, fourth]
                )
                + anticommutator(
                    pair_matrices[first, fourth], pair_matrices[second, third]
                )
            )
            / 2
        )
    return torch.stack(values)


def cayley_determinant(matrix_entries: torch.Tensor) -> torch.Tensor:
    """Brute-force row-ordered determinant over matrix-valued entries."""

    if matrix_entries.ndim != 4:
        raise ValueError("matrix_entries must have shape (n, n, d, d)")
    order, second_order, left_block, right_block = matrix_entries.shape
    if order != second_order or left_block != right_block:
        raise ValueError("matrix_entries must have shape (n, n, d, d)")
    terms = []
    for permutation in itertools.permutations(range(order)):
        inversions = sum(
            permutation[first] > permutation[second]
            for first in range(order)
            for second in range(first + 1, order)
        )
        product = torch.eye(
            left_block, dtype=matrix_entries.dtype, device=matrix_entries.device
        )
        for row, column in enumerate(permutation):
            product = product @ matrix_entries[row, column]
        terms.append((-1 if inversions % 2 else 1) * product)
    return torch.stack(terms).sum(dim=0)


def tagged_cayley_pair_data(
    matrix_entries: torch.Tensor,
    left_block_boundary: torch.Tensor,
    right_block_boundary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed a Cayley determinant into one top-degree matrix-pair amplitude.

    Shift tags make all symmetrized product orders vanish except row order.
    The returned data have ``D=N=2*n`` and ``chi=(n+1)*d``.
    """

    if matrix_entries.ndim != 4:
        raise ValueError("matrix_entries must have shape (n, n, d, d)")
    order, second_order, block, second_block = matrix_entries.shape
    if order != second_order or block != second_block or order < 1:
        raise ValueError("matrix_entries must have shape (n, n, d, d), n >= 1")
    if left_block_boundary.shape != (block,) or right_block_boundary.shape != (block,):
        raise ValueError("block boundaries must have shape (d,)")
    bond = (order + 1) * block
    dimension = 2 * order
    pair_matrices = torch.zeros(
        dimension,
        dimension,
        bond,
        bond,
        dtype=matrix_entries.dtype,
        device=matrix_entries.device,
    )
    for row in range(order):
        source = slice(row * block, (row + 1) * block)
        target = slice((row + 1) * block, (row + 2) * block)
        for column in range(order):
            pair_matrices[row, order + column, source, target] = matrix_entries[
                row, column
            ]
            pair_matrices[order + column, row, source, target] = -matrix_entries[
                row, column
            ]
    left = torch.zeros(bond, dtype=matrix_entries.dtype, device=matrix_entries.device)
    right = torch.zeros_like(left)
    left[:block] = left_block_boundary
    right[order * block :] = right_block_boundary
    return pair_matrices, left, right


def tagged_cayley_expected_amplitude(
    matrix_entries: torch.Tensor,
    left_block_boundary: torch.Tensor,
    right_block_boundary: torch.Tensor,
) -> torch.Tensor:
    """Return the top-degree amplitude predicted by the tagging reduction."""

    order = matrix_entries.shape[0]
    phase = -1 if (order * (order - 1) // 2) % 2 else 1
    return (
        phase
        * (left_block_boundary @ cayley_determinant(matrix_entries) @ right_block_boundary)
        / math.factorial(order)
    )
