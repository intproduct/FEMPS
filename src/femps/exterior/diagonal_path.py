"""Polynomial contractions for nonbranching diagonal-path FEMPS.

The state is a structured matrix-wedge FEMPS

    sum_a amplitudes[a] * u[a, :, 0] wedge ... wedge u[a, :, N-1],

where one global virtual label ``a`` is conserved at every site.  The routines
below contract the resulting ``K`` nonorthogonal Slater determinants through
``K**2`` transition pairs.  They never enumerate the nominal ``K**(N-1)``
paths of dense cores and never materialize the full particle tensor.

Column-replacement determinant formulas are used instead of overlap inverses,
so the reference production path remains valid when a bra/ket overlap matrix
is singular.  This costs extra powers of ``N`` but stays polynomial and is
fully differentiable in PyTorch.
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
        (matrix[:, :column], replacement[:, None], matrix[:, column + 1 :]),
        dim=1,
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


def diagonal_path_overlap_matrix(orbitals: torch.Tensor) -> torch.Tensor:
    """Return ``S[a,b] = <Phi_a|Phi_b>`` for all Slater paths."""

    terms, _, _ = _validate_orbitals(orbitals)
    result = torch.empty(
        (terms, terms), dtype=orbitals.dtype, device=orbitals.device
    )
    for bra in range(terms):
        bra_orbitals = orbitals[bra].conj().transpose(0, 1)
        for ket in range(terms):
            result[bra, ket] = torch.linalg.det(bra_orbitals @ orbitals[ket])
    return result


def diagonal_path_one_body_transition_matrix(
    orbitals: torch.Tensor,
    operator: torch.Tensor,
    *,
    transition_algorithm: str = "auto",
) -> torch.Tensor:
    """Return all ``<Phi_a|sum_i h(i)|Phi_b>`` transitions."""

    _validate_one_body(orbitals, operator)
    if transition_algorithm not in {"auto", "minor"}:
        raise ValueError("transition_algorithm must be auto or minor")
    terms, _, _ = orbitals.shape
    result = torch.empty(
        (terms, terms), dtype=orbitals.dtype, device=orbitals.device
    )
    acted = torch.einsum("de,ken->kdn", operator, orbitals)
    for bra in range(terms):
        bra_orbitals = orbitals[bra].conj().transpose(0, 1)
        for ket in range(terms):
            overlap = bra_orbitals @ orbitals[ket]
            insertion = bra_orbitals @ acted[ket]
            result[bra, ket] = _one_body_transition_from_projected(
                overlap,
                insertion,
                allow_inverse=transition_algorithm == "auto",
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
    if transition_algorithm not in {"auto", "minor"}:
        raise ValueError("transition_algorithm must be auto or minor")
    terms, _, particles = orbitals.shape
    result = torch.zeros(
        (terms, terms), dtype=orbitals.dtype, device=orbitals.device
    )
    if particles < 2 or left.shape[0] == 0:
        return result

    left_acted = torch.einsum("lde,ken->lkdn", left, orbitals)
    right_acted = torch.einsum("lde,ken->lkdn", right, orbitals)
    for bra in range(terms):
        bra_orbitals = orbitals[bra].conj().transpose(0, 1)
        for ket in range(terms):
            overlap = bra_orbitals @ orbitals[ket]
            value = torch.zeros((), dtype=orbitals.dtype, device=orbitals.device)
            for factor in range(left.shape[0]):
                left_insertion = bra_orbitals @ left_acted[factor, ket]
                right_insertion = bra_orbitals @ right_acted[factor, ket]
                value = value + 0.5 * weights[factor] * (
                    _mixed_transition_from_projected(
                        overlap,
                        left_insertion,
                        right_insertion,
                        allow_inverse=transition_algorithm == "auto",
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
