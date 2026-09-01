"""Small-system matrix-wedge algebra and explicit FEMPS materialization."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch

from .reference import alternating_projection, normalized_slater_from_minors


def cayley_femps_cores(
    matrix_entries: torch.Tensor,
    left_boundary: torch.Tensor,
    right_boundary: torch.Tensor,
) -> list[torch.Tensor]:
    """Embed a row-ordered Cayley determinant at fixed FEMPS bond.

    ``matrix_entries`` has shape ``(N, N, chi, chi)``. Site ``i`` emits the
    one-form ``sum_j matrix_entries[i, j] e_j``; the virtual boundaries are
    absorbed into the first and last cores. The coefficient of
    ``e_1 wedge ... wedge e_N`` is therefore

    ``left_boundary @ CDet(matrix_entries) @ right_boundary``.

    This is an exact small-system construction, not a polynomial contraction
    routine. In particular, ``chi=2`` already contains the standard hard
    noncommutative-determinant family.
    """

    if matrix_entries.ndim != 4:
        raise ValueError("matrix_entries must have shape (N, N, chi, chi)")
    particles, dimension, left_bond, right_bond = matrix_entries.shape
    if particles != dimension or left_bond != right_bond or particles < 1:
        raise ValueError("matrix_entries must have shape (N, N, chi, chi), N >= 1")
    if left_boundary.shape != (left_bond,) or right_boundary.shape != (left_bond,):
        raise ValueError("boundaries must have shape (chi,)")
    if (
        left_boundary.dtype != matrix_entries.dtype
        or right_boundary.dtype != matrix_entries.dtype
        or left_boundary.device != matrix_entries.device
        or right_boundary.device != matrix_entries.device
    ):
        raise ValueError("matrix entries and boundaries must share dtype and device")

    if particles == 1:
        value = torch.einsum(
            "a,jab,b->j", left_boundary, matrix_entries[0], right_boundary
        )
        return [value.reshape(1, 1, 1)]

    first = torch.einsum("a,jab->jb", left_boundary, matrix_entries[0])
    cores = [first.unsqueeze(0)]
    cores.extend(
        matrix_entries[site].permute(1, 0, 2)
        for site in range(1, particles - 1)
    )
    last = torch.einsum("jab,b->ja", matrix_entries[-1], right_boundary)
    cores.append(last.transpose(0, 1).unsqueeze(-1))
    return cores


def wedge_tensors(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Wedge two forms stored in the normalized antisymmetric tensor convention."""

    if left.ndim < 1 or right.ndim < 1:
        raise ValueError("wedge factors must have positive degree")
    if len(set(left.shape)) != 1 or len(set(right.shape)) != 1:
        raise ValueError("each form must use equal one-particle dimensions on all axes")
    if left.shape[0] != right.shape[0]:
        raise ValueError("wedge factors must use the same one-particle dimension")
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError("wedge factors must share dtype and device")

    left_degree, right_degree = left.ndim, right.ndim
    outer = torch.tensordot(left, right, dims=0)
    scale = math.sqrt(
        math.factorial(left_degree + right_degree)
        / (math.factorial(left_degree) * math.factorial(right_degree))
    )
    return scale * alternating_projection(outer)


def matrix_wedge(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply exterior-valued matrices using matrix contraction and wedge."""

    if left.ndim < 3 or right.ndim < 3:
        raise ValueError("matrix forms need two matrix axes and positive form degree")
    if left.shape[1] != right.shape[0]:
        raise ValueError("inner matrix dimensions do not match")
    if left.shape[2] != right.shape[2]:
        raise ValueError("one-particle dimensions do not match")
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError("matrix forms must share dtype and device")

    rows = []
    for row in range(left.shape[0]):
        columns = []
        for column in range(right.shape[1]):
            terms = [
                wedge_tensors(left[row, inner], right[inner, column])
                for inner in range(left.shape[1])
            ]
            columns.append(torch.stack(terms).sum(dim=0))
        rows.append(torch.stack(columns))
    return torch.stack(rows)


def _validate_cores(cores: Sequence[torch.Tensor]) -> tuple[int, int]:
    if not cores:
        raise ValueError("at least one FEMPS core is required")
    dimension = cores[0].shape[1] if cores[0].ndim == 3 else -1
    dtype, device = cores[0].dtype, cores[0].device
    for site, core in enumerate(cores):
        if core.ndim != 3:
            raise ValueError("FEMPS cores must have shape (left_bond, D, right_bond)")
        if core.shape[1] != dimension:
            raise ValueError("all FEMPS cores must share one-particle dimension D")
        if core.dtype != dtype or core.device != device:
            raise ValueError("all FEMPS cores must share dtype and device")
        if site and cores[site - 1].shape[2] != core.shape[0]:
            raise ValueError("neighboring FEMPS bond dimensions do not match")
    if cores[0].shape[0] != 1 or cores[-1].shape[2] != 1:
        raise ValueError("only open-boundary FEMPS with scalar boundary bonds are supported")
    if dimension < len(cores):
        raise ValueError("a nonzero N-form requires D >= N")
    return dimension, len(cores)


def materialize_femps_matrix(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Materialize a FEMPS through associative matrix-wedge multiplication."""

    _, particles = _validate_cores(cores)
    if particles == 1:
        return cores[0][0, :, 0]
    value = cores[0].permute(0, 2, 1)
    for core in cores[1:]:
        value = matrix_wedge(value, core.permute(0, 2, 1))
    return value[0, 0]


def materialize_femps_paths(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Independently materialize a FEMPS by enumerating all virtual paths."""

    dimension, particles = _validate_cores(cores)
    if particles == 1:
        return cores[0][0, :, 0]
    result = torch.zeros(
        (dimension,) * particles,
        dtype=cores[0].dtype,
        device=cores[0].device,
    )
    bond_ranges = [range(core.shape[2]) for core in cores[:-1]]
    for path in itertools.product(*bond_ranges):
        vectors = [cores[0][0, :, path[0]]]
        for site in range(1, particles - 1):
            vectors.append(cores[site][path[site - 1], :, path[site]])
        vectors.append(cores[-1][path[-1], :, 0])
        orbitals = torch.stack(vectors, dim=1)
        result = result + normalized_slater_from_minors(orbitals)
    return result


def slater_sum_cores(
    orbitals: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Embed a weighted sum of Slater terms as a diagonal-path FEMPS.

    ``orbitals`` has shape ``(terms, D, particles)``. The construction uses
    every term as one conserved virtual path, so all internal bonds equal the
    number of terms.
    """

    if orbitals.ndim != 3:
        raise ValueError("orbitals must have shape (terms, D, particles)")
    terms, dimension, particles = orbitals.shape
    if terms < 1 or particles < 1 or dimension < particles:
        raise ValueError("require terms >= 1 and D >= particles >= 1")
    if weights is None:
        weights = torch.ones(terms, dtype=orbitals.dtype, device=orbitals.device)
    if weights.shape != (terms,):
        raise ValueError("weights must have shape (terms,)")
    if weights.dtype != orbitals.dtype or weights.device != orbitals.device:
        raise ValueError("weights and orbitals must share dtype and device")
    if particles == 1:
        vector = torch.einsum("r,rdi->d", weights, orbitals)
        return [vector.reshape(1, dimension, 1)]

    cores = [orbitals[:, :, 0].transpose(0, 1).unsqueeze(0)]
    for site in range(1, particles - 1):
        diagonal = torch.diag_embed(orbitals[:, :, site].transpose(0, 1))
        cores.append(diagonal.permute(1, 0, 2))
    cores.append((weights[:, None] * orbitals[:, :, -1]).unsqueeze(-1))
    return cores


def bivector_decomposition_length(tensor: torch.Tensor) -> int:
    """Return the minimal number of decomposable terms for a two-form."""

    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("a bivector must be a square order-two tensor")
    scale = torch.linalg.vector_norm(tensor)
    tolerance = tensor.shape[0] * torch.finfo(tensor.real.dtype).eps * max(scale, 1.0)
    if torch.linalg.vector_norm(tensor + tensor.transpose(0, 1)) > tolerance:
        raise ValueError("tensor is not antisymmetric")
    rank = int(torch.linalg.matrix_rank(tensor, atol=tolerance, rtol=0.0).item())
    if rank % 2:
        raise RuntimeError("numerical skew-matrix rank should be even")
    return rank // 2
