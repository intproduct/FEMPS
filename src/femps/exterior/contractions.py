"""Exact contraction oracles for Gate A research.

All routines are differentiable PyTorch code, but their current complexity is
combinatorial in particle number, one-particle dimension, or virtual paths.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence

import torch

from .matrix_wedge import _validate_cores, materialize_femps_matrix
from .reference import normalized_slater_from_minors


def _path_orbitals(cores: Sequence[torch.Tensor]) -> Iterator[torch.Tensor]:
    _, particles = _validate_cores(cores)
    if particles == 1:
        yield cores[0][0, :, 0].unsqueeze(1)
        return
    bond_ranges = [range(core.shape[2]) for core in cores[:-1]]
    for path in itertools.product(*bond_ranges):
        vectors = [cores[0][0, :, path[0]]]
        for site in range(1, particles - 1):
            vectors.append(cores[site][path[site - 1], :, path[site]])
        vectors.append(cores[-1][path[-1], :, 0])
        yield torch.stack(vectors, dim=1)


def femps_exterior_coefficients(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Contract cores into the increasing-index basis of ``Lambda^N V``.

    The returned vector is ordered lexicographically by ``itertools.combinations``.
    It avoids virtual-path enumeration and full ``D**N`` materialization, but its
    intermediate size is ``chi_p * binom(D, p)``.
    """

    dimension, particles = _validate_cores(cores)
    coefficients = cores[0][0].transpose(0, 1)
    previous_combinations = [(index,) for index in range(dimension)]
    for site in range(1, particles):
        degree = site + 1
        next_combinations = list(itertools.combinations(range(dimension), degree))
        previous_lookup = {index: position for position, index in enumerate(previous_combinations)}
        right_bond = cores[site].shape[2]
        bond_vectors = []
        for right in range(right_bond):
            exterior_entries = []
            for target in next_combinations:
                terms = []
                for position, one_particle_index in enumerate(target):
                    previous = target[:position] + target[position + 1 :]
                    sign = -1 if (degree - 1 - position) % 2 else 1
                    terms.append(
                        sign
                        * torch.sum(
                            coefficients[:, previous_lookup[previous]]
                            * cores[site][:, one_particle_index, right]
                        )
                    )
                exterior_entries.append(torch.stack(terms).sum())
            bond_vectors.append(torch.stack(exterior_entries))
        coefficients = torch.stack(bond_vectors)
        previous_combinations = next_combinations
    return coefficients[0]


def femps_norm_exterior(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the exact norm squared from exterior coefficients."""

    coefficients = femps_exterior_coefficients(cores)
    return torch.vdot(coefficients, coefficients).real


def femps_norm_paths(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the exact norm squared as a double sum of overlap determinants."""

    paths = list(_path_orbitals(cores))
    total = torch.zeros((), dtype=cores[0].dtype, device=cores[0].device)
    for bra in paths:
        for ket in paths:
            total = total + torch.linalg.det(bra.conj().transpose(0, 1) @ ket)
    return total


def apply_one_body_sum(state: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    """Apply ``sum_i operator(i)`` to an explicit particle tensor."""

    dimension = state.shape[0]
    if operator.shape != (dimension, dimension):
        raise ValueError("one-body operator must have shape (D, D)")
    result = torch.zeros_like(state)
    for axis in range(state.ndim):
        moved = state.movedim(axis, 0)
        acted = (operator @ moved.reshape(dimension, -1)).reshape(moved.shape)
        result = result + acted.movedim(0, axis)
    return result


def one_body_expectation_explicit(
    cores: Sequence[torch.Tensor], operator: torch.Tensor
) -> torch.Tensor:
    """Evaluate an unnormalized one-body matrix element via full materialization."""

    state = materialize_femps_matrix(cores)
    return torch.vdot(state.reshape(-1), apply_one_body_sum(state, operator).reshape(-1))


def _minor_determinant(
    matrix: torch.Tensor,
    deleted_rows: tuple[int, ...],
    deleted_columns: tuple[int, ...],
) -> torch.Tensor:
    rows = [index for index in range(matrix.shape[0]) if index not in deleted_rows]
    columns = [index for index in range(matrix.shape[1]) if index not in deleted_columns]
    if not rows:
        return torch.ones((), dtype=matrix.dtype, device=matrix.device)
    return torch.linalg.det(matrix[rows][:, columns])


def one_body_expectation_paths(
    cores: Sequence[torch.Tensor], operator: torch.Tensor
) -> torch.Tensor:
    """Evaluate ``sum_i h(i)`` through generalized Slater cofactors."""

    paths = list(_path_orbitals(cores))
    particles = len(cores)
    total = torch.zeros((), dtype=cores[0].dtype, device=cores[0].device)
    for bra in paths:
        for ket in paths:
            overlap = bra.conj().transpose(0, 1) @ ket
            insertion = bra.conj().transpose(0, 1) @ operator @ ket
            for row in range(particles):
                for column in range(particles):
                    sign = -1 if (row + column) % 2 else 1
                    total = total + sign * _minor_determinant(
                        overlap, (row,), (column,)
                    ) * insertion[row, column]
    return total


def apply_two_body_sum(state: torch.Tensor, interaction: torch.Tensor) -> torch.Tensor:
    """Apply ``sum_{i<j} interaction(i,j)`` to an explicit particle tensor."""

    dimension = state.shape[0]
    if interaction.shape != (dimension,) * 4:
        raise ValueError("two-body interaction must have shape (D, D, D, D)")
    matrix = interaction.reshape(dimension**2, dimension**2)
    result = torch.zeros_like(state)
    for first in range(state.ndim):
        for second in range(first + 1, state.ndim):
            moved = state.movedim((first, second), (0, 1))
            acted = (matrix @ moved.reshape(dimension**2, -1)).reshape(moved.shape)
            result = result + acted.movedim((0, 1), (first, second))
    return result


def two_body_expectation_explicit(
    cores: Sequence[torch.Tensor], interaction: torch.Tensor
) -> torch.Tensor:
    """Evaluate an unnormalized two-body matrix element via full materialization."""

    state = materialize_femps_matrix(cores)
    acted = apply_two_body_sum(state, interaction)
    return torch.vdot(state.reshape(-1), acted.reshape(-1))


def two_body_expectation_paths(
    cores: Sequence[torch.Tensor], interaction: torch.Tensor
) -> torch.Tensor:
    """Evaluate a two-body matrix element using second-order overlap cofactors."""

    paths = list(_path_orbitals(cores))
    particles = len(cores)
    total = torch.zeros((), dtype=cores[0].dtype, device=cores[0].device)
    for bra in paths:
        for ket in paths:
            overlap = bra.conj().transpose(0, 1) @ ket
            for bra_pair in itertools.combinations(range(particles), 2):
                bra_wedge = normalized_slater_from_minors(bra[:, bra_pair])
                for ket_pair in itertools.combinations(range(particles), 2):
                    ket_wedge = normalized_slater_from_minors(ket[:, ket_pair])
                    pair_element = torch.vdot(
                        bra_wedge.reshape(-1),
                        (interaction.reshape(bra.shape[0] ** 2, -1) @ ket_wedge.reshape(-1)),
                    )
                    sign = -1 if (sum(bra_pair) + sum(ket_pair)) % 2 else 1
                    total = total + sign * _minor_determinant(
                        overlap, bra_pair, ket_pair
                    ) * pair_element
    return total


def exterior_dynamic_program_cost(
    dimension: int, bonds: Sequence[int]
) -> tuple[int, int]:
    """Return scalar multiply-add and peak coefficient counts for the DP."""

    particles = len(bonds) - 1
    if particles < 1 or len(bonds) < 2 or bonds[0] != 1 or bonds[-1] != 1:
        raise ValueError("bonds must describe a nonempty open chain")
    if dimension < particles or any(bond < 1 for bond in bonds):
        raise ValueError("require D >= N and positive bonds")
    operations = bonds[1] * dimension
    peak = bonds[1] * dimension
    for degree in range(2, particles + 1):
        entries = math.comb(dimension, degree)
        operations += bonds[degree - 1] * bonds[degree] * degree * entries
        peak = max(peak, bonds[degree] * entries)
    return operations, peak
