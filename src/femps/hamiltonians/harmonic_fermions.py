"""E1/E2 harmonic-fermion functional Hamiltonians and AGP energies."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import torch

from femps.basis import (
    harmonic_hamiltonian,
    position_matrix,
    position_squared_matrix,
)
from femps.exterior import (
    agp_one_body_transition_matrix,
    agp_overlap_matrix,
    agp_norm_generating,
    agp_one_body_expectation,
    agp_sum_norm,
    agp_sum_one_body_expectation,
    agp_sum_two_body_expectation_factorized,
    agp_two_body_expectation_factorized,
    agp_two_body_transition_matrix_factorized,
)


@dataclass(frozen=True, slots=True)
class FactorizedTwoBodyOperator:
    """Symmetrized operator-Schmidt representation of a particle-pair term."""

    left: torch.Tensor
    right: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.left.ndim != 3 or self.right.shape != self.left.shape:
            raise ValueError("left and right factors must have shape (L, D, D)")
        if self.left.shape[1] != self.left.shape[2]:
            raise ValueError("operator factors must be square")
        if self.weights.shape != (self.left.shape[0],):
            raise ValueError("weights must have shape (L,)")
        if not (
            self.left.dtype == self.right.dtype == self.weights.dtype
            and self.left.device == self.right.device == self.weights.device
        ):
            raise ValueError("all two-body tensors must share dtype and device")

    @property
    def rank(self) -> int:
        return self.left.shape[0]

    @property
    def dimension(self) -> int:
        return self.left.shape[1]

    def dense(self) -> torch.Tensor:
        """Materialize ``sum_l w_l/2 (A_l tensor B_l + B_l tensor A_l)``."""

        direct = torch.einsum("l,lpr,lqs->pqrs", self.weights, self.left, self.right)
        swapped = torch.einsum("l,lpr,lqs->pqrs", self.weights, self.right, self.left)
        return 0.5 * (direct + swapped)


def exact_noninteracting_fermion_energy(particles: int, *, omega: float = 1.0) -> float:
    """Ground energy of spinless noninteracting fermions in a 1D HO trap."""

    if particles < 1 or omega <= 0:
        raise ValueError("require particles >= 1 and omega > 0")
    return 0.5 * omega * particles * particles


def exact_interacting_pair_energy(*, kappa: float, omega: float = 1.0) -> float:
    """Ground energy for two spinless fermions with ``kappa*(x1-x2)^2/2``."""

    return exact_interacting_harmonic_fermion_energy(
        2, kappa=kappa, omega=omega
    )


def exact_interacting_harmonic_fermion_energy(
    particles: int,
    *,
    kappa: float,
    omega: float = 1.0,
) -> float:
    """Exact 1D energy for all-to-all harmonic pair interactions.

    The Hamiltonian contains ``kappa/2 * sum_{i<j}(x_i-x_j)^2``.  Its center of
    mass has frequency ``omega`` and all relative modes have frequency
    ``sqrt(omega**2 + particles*kappa)``.  Antisymmetry contributes the
    Vandermonde excitation degree ``particles*(particles-1)/2``.
    """

    if particles < 1 or omega <= 0 or omega * omega + particles * kappa <= 0:
        raise ValueError("particle count and all normal-mode frequencies must be positive")
    relative_frequency = math.sqrt(omega * omega + particles * kappa)
    return 0.5 * omega + 0.5 * (particles * particles - 1) * relative_frequency


def harmonic_pair_hamiltonian(
    basis_order: int,
    *,
    kappa: float,
    omega: float = 1.0,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, FactorizedTwoBodyOperator]:
    """Return functional matrices for the E1/E2 two-fermion Hamiltonian."""

    one_body = harmonic_hamiltonian(
        basis_order, omega=omega, dtype=dtype, device=device
    )
    position = position_matrix(basis_order, dtype=dtype, device=device)
    position_squared = position_squared_matrix(
        basis_order, dtype=dtype, device=device
    )
    identity = torch.eye(basis_order, dtype=dtype, device=device)
    left = torch.stack((position_squared, position))
    right = torch.stack((identity, position))
    weights = torch.tensor([kappa, -kappa], dtype=dtype, device=device)
    return one_body, FactorizedTwoBodyOperator(left, right, weights)


def agp_energy(
    pair_matrix: torch.Tensor,
    pairs: int,
    one_body: torch.Tensor,
    two_body: FactorizedTwoBodyOperator | None = None,
) -> torch.Tensor:
    """Return the normalized real energy of one fixed-number AGP state."""

    norm = agp_norm_generating(pair_matrix, pairs)
    if norm <= torch.finfo(norm.dtype).tiny:
        raise ValueError("AGP norm is numerically zero")
    numerator = agp_one_body_expectation(pair_matrix, pairs, one_body)
    if two_body is not None and two_body.rank:
        numerator = numerator + agp_two_body_expectation_factorized(
            pair_matrix,
            pairs,
            two_body.left,
            two_body.right,
            two_body.weights,
        )
    return (numerator / norm).real


def agp_sum_energy(
    pair_matrices: torch.Tensor,
    amplitudes: torch.Tensor,
    pairs: int,
    one_body: torch.Tensor,
    two_body: FactorizedTwoBodyOperator | None = None,
) -> torch.Tensor:
    """Return the normalized real energy of a finite AGP sum."""

    norm = agp_sum_norm(pair_matrices, amplitudes, pairs)
    if norm <= torch.finfo(norm.dtype).tiny:
        raise ValueError("AGP-sum norm is numerically zero")
    numerator = agp_sum_one_body_expectation(
        pair_matrices, amplitudes, pairs, one_body
    )
    if two_body is not None and two_body.rank:
        numerator = numerator + agp_sum_two_body_expectation_factorized(
            pair_matrices,
            amplitudes,
            pairs,
            two_body.left,
            two_body.right,
            two_body.weights,
        )
    return (numerator / norm).real


def agp_hamiltonian_transition_matrices(
    pair_matrices: torch.Tensor,
    pairs: int,
    one_body: torch.Tensor,
    two_body: FactorizedTwoBodyOperator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite-AGP overlap and Hamiltonian matrices."""

    overlap = agp_overlap_matrix(pair_matrices, pairs)
    hamiltonian = agp_one_body_transition_matrix(
        pair_matrices, pairs, one_body
    )
    if two_body is not None and two_body.rank:
        hamiltonian = hamiltonian + agp_two_body_transition_matrix_factorized(
            pair_matrices,
            pairs,
            two_body.left,
            two_body.right,
            two_body.weights,
        )
    return overlap, hamiltonian


def antisymmetric_two_particle_hamiltonian(
    one_body: torch.Tensor,
    two_body: FactorizedTwoBodyOperator | None = None,
) -> torch.Tensor:
    """Return the exact Hamiltonian in the increasing two-form basis."""

    if one_body.ndim != 2 or one_body.shape[0] != one_body.shape[1]:
        raise ValueError("one_body must be square")
    dimension = one_body.shape[0]
    identity = torch.eye(dimension, dtype=one_body.dtype, device=one_body.device)
    full = torch.kron(one_body, identity) + torch.kron(identity, one_body)
    if two_body is not None:
        if two_body.dimension != dimension:
            raise ValueError("one- and two-body dimensions do not match")
        full = full + two_body.dense().reshape(dimension**2, dimension**2)
    columns = []
    scale = math.sqrt(2.0)
    for first in range(dimension):
        for second in range(first + 1, dimension):
            vector = torch.zeros(
                dimension, dimension, dtype=one_body.dtype, device=one_body.device
            )
            vector[first, second] = 1.0 / scale
            vector[second, first] = -1.0 / scale
            columns.append(vector.reshape(-1))
    basis = torch.stack(columns, dim=1)
    return basis.conj().transpose(0, 1) @ full @ basis


def _annihilate(
    support: tuple[int, ...], orbital: int
) -> tuple[tuple[int, ...], int] | None:
    if orbital not in support:
        return None
    position = support.index(orbital)
    return support[:position] + support[position + 1 :], (-1) ** position


def _create(
    support: tuple[int, ...], orbital: int
) -> tuple[tuple[int, ...], int] | None:
    if orbital in support:
        return None
    position = sum(occupied < orbital for occupied in support)
    return (
        support[:position] + (orbital,) + support[position:],
        (-1) ** position,
    )


def _lift_one_body_to_exterior(
    operator: torch.Tensor,
    supports: list[tuple[int, ...]],
    support_index: dict[tuple[int, ...], int],
) -> torch.Tensor:
    """Lift a one-particle matrix to a fixed exterior-power sector."""

    dimension = operator.shape[0]
    lifted = torch.zeros(
        len(supports),
        len(supports),
        dtype=operator.dtype,
        device=operator.device,
    )
    for column, support in enumerate(supports):
        for annihilated_orbital in support:
            annihilated = _annihilate(support, annihilated_orbital)
            assert annihilated is not None
            intermediate, annihilation_sign = annihilated
            for created_orbital in range(dimension):
                created = _create(intermediate, created_orbital)
                if created is None:
                    continue
                final_support, creation_sign = created
                row = support_index[final_support]
                lifted[row, column] = lifted[row, column] + (
                    annihilation_sign
                    * creation_sign
                    * operator[created_orbital, annihilated_orbital]
                )
    return lifted


def antisymmetric_many_body_hamiltonian(
    one_body: torch.Tensor,
    particles: int,
    two_body: FactorizedTwoBodyOperator | None = None,
) -> torch.Tensor:
    """Build an independent exact Hamiltonian in the increasing exterior basis.

    This Slater--Condon truth path scales with ``binom(D,N)`` and is intended for
    small benchmark sectors, not production contraction.
    """

    if one_body.ndim != 2 or one_body.shape[0] != one_body.shape[1]:
        raise ValueError("one_body must be square")
    dimension = one_body.shape[0]
    if particles < 1 or particles > dimension:
        raise ValueError("require 1 <= particles <= D")
    if two_body is not None and two_body.dimension != dimension:
        raise ValueError("one- and two-body dimensions do not match")
    supports = list(itertools.combinations(range(dimension), particles))
    support_index = {support: index for index, support in enumerate(supports)}
    hamiltonian = _lift_one_body_to_exterior(
        one_body, supports, support_index
    )
    if two_body is None:
        return hamiltonian
    for term in range(two_body.rank):
        left = two_body.left[term]
        right = two_body.right[term]
        lifted_left = _lift_one_body_to_exterior(left, supports, support_index)
        lifted_right = _lift_one_body_to_exterior(right, supports, support_index)
        lifted_product = _lift_one_body_to_exterior(
            left @ right, supports, support_index
        )
        hamiltonian = hamiltonian + 0.5 * two_body.weights[term] * (
            lifted_left @ lifted_right - lifted_product
        )
    return hamiltonian


def antisymmetric_many_body_hamiltonian_dense_two_body(
    one_body: torch.Tensor,
    particles: int,
    two_body_tensor: torch.Tensor,
) -> torch.Tensor:
    """Build exact exterior truth directly from ``<pq|v|rs>`` integrals.

    This independent Slater--Condon path applies
    ``1/2 sum_pqrs V[p,q,r,s] a_p^dag a_q^dag a_s a_r``. It avoids a matrix
    product per operator-Schmidt factor and is intended only for safe sectors.
    """

    if one_body.ndim != 2 or one_body.shape[0] != one_body.shape[1]:
        raise ValueError("one_body must be square")
    dimension = one_body.shape[0]
    if two_body_tensor.shape != (dimension,) * 4:
        raise ValueError("two_body_tensor must have shape (D,D,D,D)")
    if (
        two_body_tensor.dtype != one_body.dtype
        or two_body_tensor.device != one_body.device
    ):
        raise ValueError("one- and two-body tensors must share dtype/device")
    if particles < 1 or particles > dimension:
        raise ValueError("require 1 <= particles <= D")
    supports = list(itertools.combinations(range(dimension), particles))
    support_index = {support: index for index, support in enumerate(supports)}
    hamiltonian = _lift_one_body_to_exterior(one_body, supports, support_index)
    for column, support in enumerate(supports):
        for annihilated_r in support:
            first = _annihilate(support, annihilated_r)
            assert first is not None
            after_r, sign_r = first
            for annihilated_s in after_r:
                second = _annihilate(after_r, annihilated_s)
                assert second is not None
                after_s, sign_s = second
                for created_q in range(dimension):
                    third = _create(after_s, created_q)
                    if third is None:
                        continue
                    after_q, sign_q = third
                    for created_p in range(dimension):
                        fourth = _create(after_q, created_p)
                        if fourth is None:
                            continue
                        final_support, sign_p = fourth
                        row = support_index[final_support]
                        hamiltonian[row, column] += (
                            0.5
                            * sign_r
                            * sign_s
                            * sign_q
                            * sign_p
                            * two_body_tensor[
                                created_p,
                                created_q,
                                annihilated_r,
                                annihilated_s,
                            ]
                        )
    return hamiltonian
