"""Polynomial contractions for nonbranching diagonal-path FEMPS.

The state is a structured matrix-wedge FEMPS

    sum_a amplitudes[a] * u[a, :, 0] wedge ... wedge u[a, :, N-1],

where one global virtual label ``a`` is conserved at every site.  The routines
below contract the resulting ``K`` nonorthogonal Slater determinants through
``K**2`` transition pairs.  They never enumerate the nominal ``K**(N-1)``
paths of dense cores and never materialize the full particle tensor.

The default path batches all well-conditioned transition pairs and two-body
factors. Singular pairs retain column-replacement determinant formulas, so the
production path remains valid without overlap inverses. The historical
pairwise implementation remains available as ``transition_algorithm=reference``
for value/gradient parity and performance audits.
"""

from __future__ import annotations

import itertools

import torch


def _validate_orbitals(orbitals: torch.Tensor) -> tuple[int, int, int]:
    if orbitals.ndim != 3:
        raise ValueError("orbitals must have shape (K, D, N)")
    terms, dimension, particles = orbitals.shape
    if terms < 1 or particles < 1 or dimension < particles:
        raise ValueError("require K >= 1 and D >= N >= 1")
    if not (orbitals.is_floating_point() or orbitals.is_complex()):
        raise ValueError("orbitals must use a floating or complex dtype")
    return terms, dimension, particles


def _validate_amplitudes(
    orbitals: torch.Tensor, amplitudes: torch.Tensor
) -> None:
    terms, _, _ = _validate_orbitals(orbitals)
    if amplitudes.shape != (terms,):
        raise ValueError("amplitudes must have shape (K,)")
    if amplitudes.dtype != orbitals.dtype or amplitudes.device != orbitals.device:
        raise ValueError("amplitudes and orbitals must share dtype/device")


def _validate_one_body(orbitals: torch.Tensor, operator: torch.Tensor) -> None:
    _, dimension, _ = _validate_orbitals(orbitals)
    if operator.shape != (dimension, dimension):
        raise ValueError("one-body operator must have shape (D, D)")
    if operator.dtype != orbitals.dtype or operator.device != orbitals.device:
        raise ValueError("operator and orbitals must share dtype/device")


def _validate_two_body_factors(
    orbitals: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    _, dimension, _ = _validate_orbitals(orbitals)
    if left.ndim != 3 or left.shape[1:] != (dimension, dimension):
        raise ValueError("left factors must have shape (L, D, D)")
    if right.shape != left.shape or weights.shape != (left.shape[0],):
        raise ValueError("right factors and weights must have shapes (L,D,D) and (L,)")
    tensors = (left, right, weights)
    if any(value.dtype != orbitals.dtype for value in tensors) or any(
        value.device != orbitals.device for value in tensors
    ):
        raise ValueError("two-body factors and orbitals must share dtype/device")


def _replace_column(
    matrix: torch.Tensor, replacement: torch.Tensor, column: int
) -> torch.Tensor:
    return torch.cat(
        (
            matrix[..., :column],
            replacement.unsqueeze(-1),
            matrix[..., column + 1 :],
        ),
        dim=-1,
    )


def _use_well_conditioned_path(overlap: torch.Tensor) -> bool:
    """Return whether inverse-based determinant derivatives are numerically safe."""

    with torch.no_grad():
        singular_values = torch.linalg.svdvals(overlap)
        real_dtype = singular_values.dtype
        condition_limit = torch.finfo(real_dtype).eps ** -0.5
        return bool(
            singular_values[-1] > 0
            and singular_values[0] / singular_values[-1] <= condition_limit
        )


def _well_conditioned_mask(overlaps: torch.Tensor) -> torch.Tensor:
    """Return the detached inverse-safety decision for a batch of overlaps."""

    with torch.no_grad():
        singular_values = torch.linalg.svdvals(overlaps)
        condition_limit = torch.finfo(singular_values.dtype).eps ** -0.5
        return (singular_values[..., -1] > 0) & (
            singular_values[..., 0] / singular_values[..., -1]
            <= condition_limit
        )


def _validate_transition_algorithm(transition_algorithm: str) -> None:
    if transition_algorithm not in {"auto", "minor", "reference"}:
        raise ValueError(
            "transition_algorithm must be auto, minor, or reference"
        )


def _one_body_transition_from_projected(
    overlap: torch.Tensor,
    insertion: torch.Tensor,
    *,
    allow_inverse: bool = True,
) -> torch.Tensor:
    """Return the coefficient linear in ``t`` of ``det(S + t H)``."""

    if allow_inverse and _use_well_conditioned_path(overlap):
        logarithmic_derivative = torch.linalg.solve(overlap, insertion)
        return torch.linalg.det(overlap) * torch.trace(logarithmic_derivative)
    total = torch.zeros((), dtype=overlap.dtype, device=overlap.device)
    for column in range(overlap.shape[0]):
        total = total + torch.linalg.det(
            _replace_column(overlap, insertion[:, column], column)
        )
    return total


def _mixed_transition_from_projected(
    overlap: torch.Tensor,
    left_insertion: torch.Tensor,
    right_insertion: torch.Tensor,
    *,
    allow_inverse: bool = True,
) -> torch.Tensor:
    """Return the ``t*u`` coefficient of ``det(S + t A + u B)``."""

    if allow_inverse and _use_well_conditioned_path(overlap):
        left_logarithmic = torch.linalg.solve(overlap, left_insertion)
        right_logarithmic = torch.linalg.solve(overlap, right_insertion)
        return torch.linalg.det(overlap) * (
            torch.trace(left_logarithmic) * torch.trace(right_logarithmic)
            - torch.trace(left_logarithmic @ right_logarithmic)
        )
    total = torch.zeros((), dtype=overlap.dtype, device=overlap.device)
    particles = overlap.shape[0]
    for left_column in range(particles):
        with_left = _replace_column(
            overlap, left_insertion[:, left_column], left_column
        )
        for right_column in range(particles):
            if right_column == left_column:
                continue
            total = total + torch.linalg.det(
                _replace_column(
                    with_left,
                    right_insertion[:, right_column],
                    right_column,
                )
            )
    return total


def _mixed_minor_transitions_batched_factors(
    overlap: torch.Tensor,
    left_insertions: torch.Tensor,
    right_insertions: torch.Tensor,
) -> torch.Tensor:
    """Return singular-safe mixed transitions for every factor in one batch."""

    factors, particles, _ = left_insertions.shape
    expanded_overlap = overlap.unsqueeze(0).expand(factors, -1, -1)
    total = torch.zeros(
        factors, dtype=overlap.dtype, device=overlap.device
    )
    for left_column in range(particles):
        with_left = _replace_column(
            expanded_overlap,
            left_insertions[..., left_column],
            left_column,
        )
        for right_column in range(particles):
            if right_column != left_column:
                total = total + torch.linalg.det(
                    _replace_column(
                        with_left,
                        right_insertions[..., right_column],
                        right_column,
                    )
                )
    return total


def _projected_overlap_batch(orbitals: torch.Tensor) -> torch.Tensor:
    bra_orbitals = orbitals.conj().permute(0, 2, 1)
    return torch.einsum("and,bdm->abnm", bra_orbitals, orbitals)


def _hybrid_one_body_transitions(
    overlaps: torch.Tensor, insertions: torch.Tensor
) -> torch.Tensor:
    terms, _, particles, _ = overlaps.shape
    flat_overlaps = overlaps.reshape(terms * terms, particles, particles)
    flat_insertions = insertions.reshape(
        terms * terms, particles, particles
    )
    well_mask = _well_conditioned_mask(flat_overlaps)
    well_indices = torch.nonzero(well_mask, as_tuple=False).flatten()
    fallback_indices = torch.nonzero(~well_mask, as_tuple=False).flatten()
    result = torch.zeros(
        terms * terms, dtype=overlaps.dtype, device=overlaps.device
    )
    if well_indices.numel():
        selected_overlaps = flat_overlaps[well_indices]
        logarithmic_derivatives = torch.linalg.solve(
            selected_overlaps, flat_insertions[well_indices]
        )
        values = torch.linalg.det(selected_overlaps) * torch.diagonal(
            logarithmic_derivatives, dim1=-2, dim2=-1
        ).sum(-1)
        result = result.index_copy(0, well_indices, values)
    if fallback_indices.numel():
        values = torch.stack(
            [
                _one_body_transition_from_projected(
                    flat_overlaps[index],
                    flat_insertions[index],
                    allow_inverse=False,
                )
                for index in fallback_indices.tolist()
            ]
        )
        result = result.index_copy(0, fallback_indices, values)
    return result.reshape(terms, terms)


def _hybrid_two_body_transitions(
    overlaps: torch.Tensor,
    left_insertions: torch.Tensor,
    right_insertions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    terms, _, factors, particles, _ = left_insertions.shape
    pairs = terms * terms
    flat_overlaps = overlaps.reshape(pairs, particles, particles)
    flat_left = left_insertions.reshape(pairs, factors, particles, particles)
    flat_right = right_insertions.reshape(pairs, factors, particles, particles)
    well_mask = _well_conditioned_mask(flat_overlaps)
    well_indices = torch.nonzero(well_mask, as_tuple=False).flatten()
    fallback_indices = torch.nonzero(~well_mask, as_tuple=False).flatten()
    result = torch.zeros(pairs, dtype=overlaps.dtype, device=overlaps.device)
    if well_indices.numel():
        selected_overlaps = flat_overlaps[well_indices]
        left_logarithmic = torch.linalg.solve(
            selected_overlaps[:, None], flat_left[well_indices]
        )
        right_logarithmic = torch.linalg.solve(
            selected_overlaps[:, None], flat_right[well_indices]
        )
        left_trace = torch.diagonal(
            left_logarithmic, dim1=-2, dim2=-1
        ).sum(-1)
        right_trace = torch.diagonal(
            right_logarithmic, dim1=-2, dim2=-1
        ).sum(-1)
        mixed_trace = torch.einsum(
            "plij,plji->pl", left_logarithmic, right_logarithmic
        )
        per_factor = torch.linalg.det(selected_overlaps)[:, None] * (
            left_trace * right_trace - mixed_trace
        )
        values = 0.5 * torch.sum(per_factor * weights[None], dim=-1)
        result = result.index_copy(0, well_indices, values)
    if fallback_indices.numel():
        values = torch.stack(
            [
                0.5
                * torch.sum(
                    weights
                    * _mixed_minor_transitions_batched_factors(
                        flat_overlaps[index],
                        flat_left[index],
                        flat_right[index],
                    )
                )
                for index in fallback_indices.tolist()
            ]
        )
        result = result.index_copy(0, fallback_indices, values)
    return result.reshape(terms, terms)


def diagonal_path_overlap_matrix(orbitals: torch.Tensor) -> torch.Tensor:
    """Return ``S[a,b] = <Phi_a|Phi_b>`` for all Slater paths."""

    _validate_orbitals(orbitals)
    return torch.linalg.det(_projected_overlap_batch(orbitals))


def diagonal_path_one_body_transition_matrix(
    orbitals: torch.Tensor,
    operator: torch.Tensor,
    *,
    transition_algorithm: str = "auto",
) -> torch.Tensor:
    """Return all ``<Phi_a|sum_i h(i)|Phi_b>`` transitions."""

    _validate_one_body(orbitals, operator)
    _validate_transition_algorithm(transition_algorithm)
    terms, _, _ = orbitals.shape
    acted = torch.einsum("de,ken->kdn", operator, orbitals)
    bra_orbitals = orbitals.conj().permute(0, 2, 1)
    overlaps = torch.einsum("and,bdm->abnm", bra_orbitals, orbitals)
    insertions = torch.einsum("and,bdm->abnm", bra_orbitals, acted)
    if transition_algorithm == "auto":
        return _hybrid_one_body_transitions(overlaps, insertions)
    result = torch.empty(
        (terms, terms), dtype=orbitals.dtype, device=orbitals.device
    )
    for bra in range(terms):
        for ket in range(terms):
            result[bra, ket] = _one_body_transition_from_projected(
                overlaps[bra, ket],
                insertions[bra, ket],
                allow_inverse=transition_algorithm == "reference",
            )
    return result


def diagonal_path_two_body_transition_matrix_factorized(
    orbitals: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    weights: torch.Tensor,
    *,
    transition_algorithm: str = "auto",
) -> torch.Tensor:
    """Return transitions for a factorized symmetric pair Hamiltonian.

    The operator convention is

    ``sum_l weights[l]/2 * sum_{i<j}(L_l(i)R_l(j)+R_l(i)L_l(j))``.
    """

    _validate_two_body_factors(orbitals, left, right, weights)
    _validate_transition_algorithm(transition_algorithm)
    terms, _, particles = orbitals.shape
    result = torch.zeros(
        (terms, terms), dtype=orbitals.dtype, device=orbitals.device
    )
    if particles < 2 or left.shape[0] == 0:
        return result

    left_acted = torch.einsum("lde,ken->lkdn", left, orbitals)
    right_acted = torch.einsum("lde,ken->lkdn", right, orbitals)
    bra_orbitals = orbitals.conj().permute(0, 2, 1)
    overlaps = torch.einsum("and,bdm->abnm", bra_orbitals, orbitals)
    left_insertions = torch.einsum(
        "and,lbdm->ablnm", bra_orbitals, left_acted
    )
    right_insertions = torch.einsum(
        "and,lbdm->ablnm", bra_orbitals, right_acted
    )
    if transition_algorithm == "auto":
        return _hybrid_two_body_transitions(
            overlaps, left_insertions, right_insertions, weights
        )
    for bra in range(terms):
        for ket in range(terms):
            value = torch.zeros((), dtype=orbitals.dtype, device=orbitals.device)
            for factor in range(left.shape[0]):
                value = value + 0.5 * weights[factor] * (
                    _mixed_transition_from_projected(
                        overlaps[bra, ket],
                        left_insertions[bra, ket, factor],
                        right_insertions[bra, ket, factor],
                        allow_inverse=transition_algorithm == "reference",
                    )
                )
            result[bra, ket] = value
    return result


def diagonal_path_hamiltonian_matrices(
    orbitals: torch.Tensor,
    one_body: torch.Tensor,
    *,
    two_body_left: torch.Tensor | None = None,
    two_body_right: torch.Tensor | None = None,
    two_body_weights: torch.Tensor | None = None,
    transition_algorithm: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return overlap and Hamiltonian transition matrices."""

    overlap = diagonal_path_overlap_matrix(orbitals)
    hamiltonian = diagonal_path_one_body_transition_matrix(
        orbitals, one_body, transition_algorithm=transition_algorithm
    )
    supplied = (
        two_body_left is not None,
        two_body_right is not None,
        two_body_weights is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError("supply all two-body factor tensors or none")
    if all(supplied):
        assert two_body_left is not None
        assert two_body_right is not None
        assert two_body_weights is not None
        hamiltonian = hamiltonian + (
            diagonal_path_two_body_transition_matrix_factorized(
                orbitals,
                two_body_left,
                two_body_right,
                two_body_weights,
                transition_algorithm=transition_algorithm,
            )
        )
    return overlap, hamiltonian


def diagonal_path_norm(
    orbitals: torch.Tensor, amplitudes: torch.Tensor
) -> torch.Tensor:
    """Return the exact squared norm of a diagonal-path FEMPS."""

    _validate_amplitudes(orbitals, amplitudes)
    overlap = diagonal_path_overlap_matrix(orbitals)
    return torch.vdot(amplitudes, overlap @ amplitudes).real


def diagonal_path_energy(
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    one_body: torch.Tensor,
    *,
    two_body_left: torch.Tensor | None = None,
    two_body_right: torch.Tensor | None = None,
    two_body_weights: torch.Tensor | None = None,
    transition_algorithm: str = "auto",
) -> torch.Tensor:
    """Return the normalized exact energy of a diagonal-path FEMPS."""

    _validate_amplitudes(orbitals, amplitudes)
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        orbitals,
        one_body,
        two_body_left=two_body_left,
        two_body_right=two_body_right,
        two_body_weights=two_body_weights,
        transition_algorithm=transition_algorithm,
    )
    norm = torch.vdot(amplitudes, overlap @ amplitudes).real
    if norm <= torch.finfo(norm.dtype).tiny:
        raise ValueError("diagonal-path FEMPS norm is numerically zero")
    numerator = torch.vdot(amplitudes, hamiltonian @ amplitudes)
    return (numerator / norm).real


def diagonal_path_exterior_coefficients(
    orbitals: torch.Tensor, amplitudes: torch.Tensor
) -> torch.Tensor:
    """Materialize increasing-basis coefficients for bounded truth checks."""

    _validate_amplitudes(orbitals, amplitudes)
    _, dimension, particles = orbitals.shape
    coefficients = []
    for support in itertools.combinations(range(dimension), particles):
        minors = torch.linalg.det(orbitals[:, support, :])
        coefficients.append(torch.dot(amplitudes, minors))
    return torch.stack(coefficients)


def diagonal_path_structural_counts(
    particles: int, dimension: int, terms: int, factor_rank: int = 0
) -> dict[str, int]:
    """Return exact structural counts for the singular-safe reference path."""

    if min(particles, dimension, terms) < 1 or dimension < particles:
        raise ValueError("require D >= N >= 1 and K >= 1")
    if factor_rank < 0:
        raise ValueError("factor_rank must be nonnegative")
    return {
        "stored_orbital_scalars": terms * dimension * particles,
        "stored_amplitude_scalars": terms,
        "transition_pairs": terms * terms,
        "one_body_determinants": terms * terms * particles,
        "two_body_determinants": (
            terms * terms * factor_rank * particles * (particles - 1)
        ),
        "enumerated_virtual_paths": 0,
        "materialized_particle_coefficients": 0,
    }


def diagonal_path_transition_diagnostics(orbitals: torch.Tensor) -> dict[str, int]:
    """Count inverse-fast and singular-safe transition pairs for a state."""

    terms, _, _ = _validate_orbitals(orbitals)
    fast_pairs = 0
    fallback_pairs = 0
    for bra in range(terms):
        bra_orbitals = orbitals[bra].conj().transpose(0, 1)
        for ket in range(terms):
            overlap = bra_orbitals @ orbitals[ket]
            if _use_well_conditioned_path(overlap):
                fast_pairs += 1
            else:
                fallback_pairs += 1
    return {
        "transition_pairs": terms * terms,
        "well_conditioned_inverse_pairs": fast_pairs,
        "singular_safe_minor_pairs": fallback_pairs,
    }
