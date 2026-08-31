"""Fixed-number Pfaffian/AGP states as a structured FEMPS subclass.

For an antisymmetric pair matrix ``F``, the ``2M``-particle state has exterior
coefficients ``pf(F[I, I])``. It equals ``Omega**M / M!`` and admits both a
bond-structured FEMPS representation and polynomial overlap contractions.
"""

from __future__ import annotations

import itertools
import math

import torch


def _validate_pair_matrix(pair_matrix: torch.Tensor, pairs: int | None = None) -> int:
    if pair_matrix.ndim != 2 or pair_matrix.shape[0] != pair_matrix.shape[1]:
        raise ValueError("pair_matrix must be a square matrix")
    dimension = pair_matrix.shape[0]
    if pairs is not None and (pairs < 1 or 2 * pairs > dimension):
        raise ValueError("require 1 <= pairs <= D/2")
    scale = torch.linalg.vector_norm(pair_matrix)
    tolerance = dimension * torch.finfo(pair_matrix.real.dtype).eps * max(
        float(scale.detach().cpu()), 1.0
    )
    if torch.linalg.vector_norm(pair_matrix + pair_matrix.transpose(0, 1)) > tolerance:
        raise ValueError("pair_matrix must be antisymmetric under transpose")
    return dimension


def pair_matrix_from_channels(
    left_orbitals: torch.Tensor,
    right_orbitals: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``sum_a w_a (u_a v_a^T - v_a u_a^T)``.

    Orbital arrays have shape ``(channels, D)``. Transpose, rather than
    Hermitian transpose, is required because these are exterior coefficients.
    """

    if left_orbitals.ndim != 2 or right_orbitals.shape != left_orbitals.shape:
        raise ValueError("left_orbitals and right_orbitals must have shape (channels, D)")
    if left_orbitals.dtype != right_orbitals.dtype or left_orbitals.device != right_orbitals.device:
        raise ValueError("orbital arrays must share dtype and device")
    channels = left_orbitals.shape[0]
    if channels < 1:
        raise ValueError("at least one pair channel is required")
    if weights is None:
        weights = torch.ones(
            channels, dtype=left_orbitals.dtype, device=left_orbitals.device
        )
    if weights.shape != (channels,):
        raise ValueError("weights must have shape (channels,)")
    if weights.dtype != left_orbitals.dtype or weights.device != left_orbitals.device:
        raise ValueError("weights and orbitals must share dtype and device")
    weighted_left = weights[:, None] * left_orbitals
    return weighted_left.transpose(0, 1) @ right_orbitals - right_orbitals.transpose(
        0, 1
    ) @ weighted_left


def real_skew_pair_decomposition(
    pair_matrix: torch.Tensor,
    channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily decompose a real skew matrix into orthonormal pair channels.

    The routine is a deterministic reference based on successive largest
    singular two-planes. Complex matrices with a material imaginary part are
    rejected; a general complex skew-Takagi implementation is future work.
    """

    dimension = _validate_pair_matrix(pair_matrix)
    maximum_channels = dimension // 2
    if channels is None:
        channels = maximum_channels
    if not 1 <= channels <= maximum_channels:
        raise ValueError("channels must satisfy 1 <= channels <= floor(D/2)")
    imaginary_scale = (
        float(torch.linalg.vector_norm(pair_matrix.imag).detach().cpu())
        if pair_matrix.is_complex()
        else 0.0
    )
    real_scale = max(
        float(torch.linalg.vector_norm(pair_matrix.real).detach().cpu()), 1.0
    )
    if imaginary_scale > 1e-12 * real_scale:
        raise ValueError("real_skew_pair_decomposition requires a real pair matrix")
    residual = pair_matrix.real.clone()
    left_vectors = []
    right_vectors = []
    weights = []
    for _ in range(channels):
        left_singular, singular_values, _ = torch.linalg.svd(
            residual, full_matrices=False
        )
        weight = singular_values[0]
        if weight <= 10 * torch.finfo(weight.dtype).eps * real_scale:
            left = torch.zeros(dimension, dtype=residual.dtype, device=residual.device)
            right = torch.zeros_like(left)
        else:
            left = left_singular[:, 0]
            right = residual.transpose(0, 1) @ left / weight
            residual = residual - weight * (
                torch.outer(left, right) - torch.outer(right, left)
            )
        left_vectors.append(left)
        right_vectors.append(right)
        weights.append(weight)
    output_dtype = pair_matrix.dtype
    return (
        torch.stack(left_vectors).to(output_dtype),
        torch.stack(right_vectors).to(output_dtype),
        torch.stack(weights).to(output_dtype),
    )


def pfaffian_recursive(matrix: torch.Tensor) -> torch.Tensor:
    """Differentiable small-system Pfaffian reference using Laplace recursion."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Pfaffian input must be square")
    order = matrix.shape[0]
    if order % 2:
        raise ValueError("Pfaffian input order must be even")
    if order == 0:
        return torch.ones((), dtype=matrix.dtype, device=matrix.device)
    if order == 2:
        return matrix[0, 1]
    terms = []
    for column in range(1, order):
        retained = [index for index in range(1, order) if index != column]
        minor = matrix[retained][:, retained]
        sign = -1 if (column + 1) % 2 else 1
        terms.append(sign * matrix[0, column] * pfaffian_recursive(minor))
    return torch.stack(terms).sum()


def agp_exterior_coefficients(pair_matrix: torch.Tensor, pairs: int) -> torch.Tensor:
    """Return lexicographically ordered coefficients of ``Omega^pairs/pairs!``."""

    dimension = _validate_pair_matrix(pair_matrix, pairs)
    coefficients = [
        pfaffian_recursive(pair_matrix[list(support)][:, list(support)])
        for support in itertools.combinations(range(dimension), 2 * pairs)
    ]
    return torch.stack(coefficients)


def agp_tensor(pair_matrix: torch.Tensor, pairs: int) -> torch.Tensor:
    """Materialize the normalized-exterior-convention particle tensor."""

    dimension = _validate_pair_matrix(pair_matrix, pairs)
    particles = 2 * pairs
    result = torch.zeros(
        (dimension,) * particles,
        dtype=pair_matrix.dtype,
        device=pair_matrix.device,
    )
    scale = math.sqrt(math.factorial(particles))
    permutations = list(itertools.permutations(range(particles)))
    for support in itertools.combinations(range(dimension), particles):
        coefficient = pfaffian_recursive(pair_matrix[list(support)][:, list(support)])
        for permutation in permutations:
            inversions = sum(
                permutation[first] > permutation[second]
                for first in range(particles)
                for second in range(first + 1, particles)
            )
            sign = -1 if inversions % 2 else 1
            index = tuple(support[position] for position in permutation)
            result[index] = sign * coefficient / scale
    return result


def agp_femps_cores(
    left_orbitals: torch.Tensor,
    right_orbitals: torch.Tensor,
    pairs: int,
    weights: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Embed a pair-power state into a strictly ordered-channel FEMPS.

    All internal bonds equal the number of pair channels. Virtual paths choose
    ``pairs`` strictly increasing channels, so there are ``binom(r, pairs)``
    nonzero paths rather than ``r**(2*pairs-1)`` unrestricted paths.
    """

    if left_orbitals.ndim != 2 or right_orbitals.shape != left_orbitals.shape:
        raise ValueError("orbital arrays must have shape (channels, D)")
    channels, dimension = left_orbitals.shape
    if pairs < 1 or pairs > channels or 2 * pairs > dimension:
        raise ValueError("require 1 <= pairs <= channels and 2*pairs <= D")
    if left_orbitals.dtype != right_orbitals.dtype or left_orbitals.device != right_orbitals.device:
        raise ValueError("orbital arrays must share dtype and device")
    if weights is None:
        weights = torch.ones(
            channels, dtype=left_orbitals.dtype, device=left_orbitals.device
        )
    if weights.shape != (channels,) or weights.dtype != left_orbitals.dtype or weights.device != left_orbitals.device:
        raise ValueError("weights must have shape (channels,) and share dtype/device")

    selected_left = weights[:, None] * left_orbitals
    cores = [selected_left.transpose(0, 1).unsqueeze(0)]
    if pairs == 1:
        cores.append(right_orbitals.unsqueeze(-1))
        return cores

    diagonal_right = torch.diag_embed(right_orbitals.transpose(0, 1)).permute(1, 0, 2)
    upper_mask = torch.triu(
        torch.ones(
            channels,
            channels,
            dtype=left_orbitals.real.dtype,
            device=left_orbitals.device,
        ),
        diagonal=1,
    ).to(left_orbitals.dtype)
    transition_left = torch.einsum("ab,bd->adb", upper_mask, selected_left)
    for pair_index in range(pairs):
        if pair_index == 0:
            cores.append(diagonal_right)
            continue
        cores.append(transition_left)
        if pair_index == pairs - 1:
            cores.append(right_orbitals.unsqueeze(-1))
        else:
            cores.append(diagonal_right)
    return cores


def blocked_agp_femps_cores(
    blocked_orbital: torch.Tensor,
    left_orbitals: torch.Tensor,
    right_orbitals: torch.Tensor,
    pairs: int,
    weights: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Return FEMPS cores for ``blocked_orbital wedge Psi_pairs(F)``."""

    if blocked_orbital.ndim != 1:
        raise ValueError("blocked_orbital must have shape (D,)")
    if left_orbitals.ndim != 2 or right_orbitals.shape != left_orbitals.shape:
        raise ValueError("orbital arrays must have shape (channels, D)")
    if left_orbitals.shape[1] != blocked_orbital.shape[0]:
        raise ValueError("blocked orbital and pair channels must share D")
    if (
        blocked_orbital.dtype != left_orbitals.dtype
        or blocked_orbital.device != left_orbitals.device
    ):
        raise ValueError("blocked orbital and pair channels must share dtype/device")
    blocked_core = blocked_orbital.reshape(1, -1, 1)
    if pairs == 0:
        return [blocked_core]
    return [blocked_core] + agp_femps_cores(
        left_orbitals, right_orbitals, pairs, weights
    )


def _detached_max_abs_scale(matrix: torch.Tensor) -> torch.Tensor:
    """Return a finite nonzero homogeneous scale without differentiating it."""

    maximum = torch.amax(torch.abs(matrix)).detach()
    one = torch.ones((), dtype=maximum.dtype, device=maximum.device)
    return torch.where(maximum > torch.finfo(maximum.dtype).tiny, maximum, one)


def _overlap_recurrence(
    bra_pair_matrix: torch.Tensor,
    ket_pair_matrix: torch.Tensor,
    pairs: int,
    ket_direction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dimension = _validate_pair_matrix(bra_pair_matrix, pairs)
    if ket_pair_matrix.shape != (dimension, dimension):
        raise ValueError("bra and ket pair matrices must have the same shape")
    _validate_pair_matrix(ket_pair_matrix, pairs)
    if bra_pair_matrix.dtype != ket_pair_matrix.dtype or bra_pair_matrix.device != ket_pair_matrix.device:
        raise ValueError("bra and ket pair matrices must share dtype and device")
    if ket_direction is not None and ket_direction.shape != ket_pair_matrix.shape:
        raise ValueError("ket_direction must match the pair-matrix shape")

    # Newton traces contain powers of F^dagger G.  Homogeneous input scaling
    # keeps those powers near unit magnitude without changing either the value
    # or its AD derivative; the detached scales are restored at the end.
    bra_scale = _detached_max_abs_scale(bra_pair_matrix)
    ket_scale = _detached_max_abs_scale(ket_pair_matrix)
    normalized_bra = bra_pair_matrix / bra_scale
    normalized_ket = ket_pair_matrix / ket_scale
    overlap_matrix = normalized_bra.conj().transpose(0, 1) @ normalized_ket
    direction_matrix = (
        None
        if ket_direction is None
        else normalized_bra.conj().transpose(0, 1)
        @ (ket_direction / ket_scale)
    )
    power_matrix = torch.eye(
        dimension, dtype=overlap_matrix.dtype, device=overlap_matrix.device
    )
    traces = [
        torch.zeros((), dtype=overlap_matrix.dtype, device=overlap_matrix.device)
    ]
    trace_derivatives = (
        [torch.zeros((), dtype=overlap_matrix.dtype, device=overlap_matrix.device)]
        if direction_matrix is not None
        else None
    )
    for power in range(1, pairs + 1):
        if trace_derivatives is not None and direction_matrix is not None:
            trace_derivatives.append(
                power
                * torch.sum(
                    power_matrix * direction_matrix.transpose(0, 1)
                )
            )
        power_matrix = power_matrix @ overlap_matrix
        traces.append(torch.trace(power_matrix))

    coefficients = [
        torch.ones((), dtype=overlap_matrix.dtype, device=overlap_matrix.device)
    ]
    derivatives = (
        [torch.zeros((), dtype=overlap_matrix.dtype, device=overlap_matrix.device)]
        if direction_matrix is not None
        else None
    )
    for degree in range(1, pairs + 1):
        value_terms = []
        derivative_terms = []
        for power in range(1, degree + 1):
            sign = -1 if (power + 1) % 2 else 1
            trace = traces[power]
            value_terms.append(0.5 * sign * trace * coefficients[degree - power])
            if derivatives is not None and direction_matrix is not None:
                assert trace_derivatives is not None
                trace_derivative = trace_derivatives[power]
                derivative_terms.append(
                    0.5
                    * sign
                    * (
                        trace_derivative * coefficients[degree - power]
                        + trace * derivatives[degree - power]
                    )
                )
        coefficients.append(torch.stack(value_terms).sum() / degree)
        if derivatives is not None:
            derivatives.append(torch.stack(derivative_terms).sum() / degree)
    output_scale = (bra_scale * ket_scale).pow(pairs)
    return (
        output_scale * coefficients[pairs],
        None if derivatives is None else output_scale * derivatives[pairs],
    )


def agp_overlap_generating(
    bra_pair_matrix: torch.Tensor,
    ket_pair_matrix: torch.Tensor,
    pairs: int,
) -> torch.Tensor:
    """Return the fixed-pair overlap in ``O(pairs * D^3)`` time."""

    overlap, _ = _overlap_recurrence(bra_pair_matrix, ket_pair_matrix, pairs)
    return overlap


def agp_norm_generating(pair_matrix: torch.Tensor, pairs: int) -> torch.Tensor:
    """Return the fixed-pair norm using a positive singular-value recurrence.

    A complex skew matrix has pairwise-degenerate singular values.  If their
    squared pair averages are ``lambda_a``, the norm is the degree-``pairs``
    elementary symmetric polynomial in the ``lambda_a``.  This avoids the
    catastrophic alternating-trace cancellation of a generic Newton recurrence.
    """

    dimension = _validate_pair_matrix(pair_matrix, pairs)
    matrix_scale = _detached_max_abs_scale(pair_matrix)
    singular_values = torch.linalg.svdvals(pair_matrix / matrix_scale)
    paired_count = 2 * (dimension // 2)
    pair_strengths = singular_values[:paired_count].reshape(-1, 2).square().mean(1)
    coefficients = torch.zeros(
        pairs + 1,
        dtype=pair_matrix.real.dtype,
        device=pair_matrix.device,
    )
    coefficients[0] = 1
    for strength in pair_strengths:
        coefficients = torch.cat(
            (
                coefficients[:1],
                coefficients[1:] + strength * coefficients[:-1],
            )
        )
    return coefficients[pairs] * matrix_scale.pow(2 * pairs)


def agp_log_norm(pair_matrix: torch.Tensor, pairs: int) -> torch.Tensor:
    """Return the logarithm of the fixed-pair norm without overflow/underflow."""

    dimension = _validate_pair_matrix(pair_matrix, pairs)
    matrix_scale = _detached_max_abs_scale(pair_matrix)
    singular_values = torch.linalg.svdvals(pair_matrix / matrix_scale)
    paired_count = 2 * (dimension // 2)
    pair_strengths = singular_values[:paired_count].reshape(-1, 2).square().mean(1)
    negative_infinity = torch.full(
        (),
        -torch.inf,
        dtype=pair_matrix.real.dtype,
        device=pair_matrix.device,
    )
    log_coefficients = [
        torch.zeros((), dtype=pair_matrix.real.dtype, device=pair_matrix.device)
    ] + [negative_infinity for _ in range(pairs)]
    for strength in pair_strengths:
        log_strength = torch.log(strength)
        updated = [log_coefficients[0]]
        for degree in range(1, pairs + 1):
            updated.append(
                torch.logaddexp(
                    log_coefficients[degree],
                    log_strength + log_coefficients[degree - 1],
                )
            )
        log_coefficients = updated
    return log_coefficients[pairs] + 2 * pairs * torch.log(matrix_scale)


def _validate_blocked_state(
    pair_matrix: torch.Tensor,
    blocked_orbital: torch.Tensor,
    pairs: int,
) -> int:
    dimension = _validate_pair_matrix(pair_matrix)
    if blocked_orbital.shape != (dimension,):
        raise ValueError("blocked_orbital must have shape (D,)")
    if (
        blocked_orbital.dtype != pair_matrix.dtype
        or blocked_orbital.device != pair_matrix.device
    ):
        raise ValueError("blocked orbital and pair matrix must share dtype/device")
    if pairs < 0 or 2 * pairs + 1 > dimension:
        raise ValueError("require 0 <= pairs and 2*pairs+1 <= D")
    return dimension


def blocked_augmented_pair_matrix(
    pair_matrix: torch.Tensor,
    blocked_orbital: torch.Tensor,
    pairs: int,
) -> torch.Tensor:
    """Append an auxiliary orbital coupled only to the blocked orbital."""

    dimension = _validate_blocked_state(pair_matrix, blocked_orbital, pairs)
    augmented = torch.zeros(
        dimension + 1,
        dimension + 1,
        dtype=pair_matrix.dtype,
        device=pair_matrix.device,
    )
    augmented[:dimension, :dimension] = pair_matrix
    augmented[:dimension, dimension] = blocked_orbital
    augmented[dimension, :dimension] = -blocked_orbital
    return augmented


def blocked_agp_exterior_coefficients(
    pair_matrix: torch.Tensor,
    blocked_orbital: torch.Tensor,
    pairs: int,
) -> torch.Tensor:
    """Return coefficients of ``u wedge Omega_F**pairs / pairs!``."""

    dimension = _validate_blocked_state(pair_matrix, blocked_orbital, pairs)
    particles = 2 * pairs + 1
    coefficients = []
    for support in itertools.combinations(range(dimension), particles):
        indices = list(support)
        restricted_pair = pair_matrix[indices][:, indices]
        restricted_block = blocked_orbital[indices]
        augmented = torch.zeros(
            particles + 1,
            particles + 1,
            dtype=pair_matrix.dtype,
            device=pair_matrix.device,
        )
        augmented[:particles, :particles] = restricted_pair
        augmented[:particles, particles] = restricted_block
        augmented[particles, :particles] = -restricted_block
        coefficients.append(pfaffian_recursive(augmented))
    return torch.stack(coefficients)


def blocked_agp_tensor(
    pair_matrix: torch.Tensor,
    blocked_orbital: torch.Tensor,
    pairs: int,
) -> torch.Tensor:
    """Materialize a blocked AGP in the normalized exterior convention."""

    dimension = _validate_blocked_state(pair_matrix, blocked_orbital, pairs)
    particles = 2 * pairs + 1
    coefficients = blocked_agp_exterior_coefficients(
        pair_matrix, blocked_orbital, pairs
    )
    result = torch.zeros(
        (dimension,) * particles,
        dtype=pair_matrix.dtype,
        device=pair_matrix.device,
    )
    scale = math.sqrt(math.factorial(particles))
    permutations = list(itertools.permutations(range(particles)))
    for coefficient, support in zip(
        coefficients, itertools.combinations(range(dimension), particles)
    ):
        for permutation in permutations:
            inversions = sum(
                permutation[first] > permutation[second]
                for first in range(particles)
                for second in range(first + 1, particles)
            )
            sign = -1 if inversions % 2 else 1
            index = tuple(support[position] for position in permutation)
            result[index] = sign * coefficient / scale
    return result


def _blocked_physical_even_overlap(
    bra_pair_matrix: torch.Tensor,
    ket_pair_matrix: torch.Tensor,
    pairs: int,
) -> torch.Tensor:
    if 2 * (pairs + 1) > bra_pair_matrix.shape[0]:
        return torch.zeros(
            (), dtype=bra_pair_matrix.dtype, device=bra_pair_matrix.device
        )
    return agp_overlap_generating(
        bra_pair_matrix, ket_pair_matrix, pairs + 1
    )


def blocked_agp_overlap(
    bra_pair_matrix: torch.Tensor,
    bra_blocked_orbital: torch.Tensor,
    ket_pair_matrix: torch.Tensor,
    ket_blocked_orbital: torch.Tensor,
    pairs: int,
) -> torch.Tensor:
    """Return a blocked-AGP overlap via auxiliary-sector subtraction."""

    dimension = _validate_blocked_state(
        bra_pair_matrix, bra_blocked_orbital, pairs
    )
    if ket_pair_matrix.shape != (dimension, dimension):
        raise ValueError("bra and ket pair matrices must have the same shape")
    _validate_blocked_state(ket_pair_matrix, ket_blocked_orbital, pairs)
    augmented_bra = blocked_augmented_pair_matrix(
        bra_pair_matrix, bra_blocked_orbital, pairs
    )
    augmented_ket = blocked_augmented_pair_matrix(
        ket_pair_matrix, ket_blocked_orbital, pairs
    )
    return agp_overlap_generating(augmented_bra, augmented_ket, pairs + 1) - (
        _blocked_physical_even_overlap(bra_pair_matrix, ket_pair_matrix, pairs)
    )


def blocked_agp_norm(
    pair_matrix: torch.Tensor,
    blocked_orbital: torch.Tensor,
    pairs: int,
) -> torch.Tensor:
    """Return the norm of a blocked AGP state."""

    _validate_blocked_state(pair_matrix, blocked_orbital, pairs)
    augmented = blocked_augmented_pair_matrix(
        pair_matrix, blocked_orbital, pairs
    )
    result = agp_norm_generating(augmented, pairs + 1)
    if 2 * (pairs + 1) <= pair_matrix.shape[0]:
        result = result - agp_norm_generating(pair_matrix, pairs + 1)
    return result.real


def _extend_blocked_operator(operator: torch.Tensor) -> torch.Tensor:
    dimension = operator.shape[0]
    extended = torch.zeros(
        dimension + 1,
        dimension + 1,
        dtype=operator.dtype,
        device=operator.device,
    )
    extended[:dimension, :dimension] = operator
    return extended


def blocked_agp_one_body_transition(
    bra_pair_matrix: torch.Tensor,
    bra_blocked_orbital: torch.Tensor,
    ket_pair_matrix: torch.Tensor,
    ket_blocked_orbital: torch.Tensor,
    pairs: int,
    operator: torch.Tensor,
) -> torch.Tensor:
    """Return a one-body transition element between blocked AGPs."""

    dimension = _validate_blocked_state(
        bra_pair_matrix, bra_blocked_orbital, pairs
    )
    _validate_blocked_state(ket_pair_matrix, ket_blocked_orbital, pairs)
    if operator.shape != (dimension, dimension):
        raise ValueError("operator must have shape (D, D)")
    augmented_bra = blocked_augmented_pair_matrix(
        bra_pair_matrix, bra_blocked_orbital, pairs
    )
    augmented_ket = blocked_augmented_pair_matrix(
        ket_pair_matrix, ket_blocked_orbital, pairs
    )
    result = agp_one_body_transition(
        augmented_bra,
        augmented_ket,
        pairs + 1,
        _extend_blocked_operator(operator),
    )
    if 2 * (pairs + 1) <= dimension:
        result = result - agp_one_body_transition(
            bra_pair_matrix, ket_pair_matrix, pairs + 1, operator
        )
    return result


def blocked_agp_one_body_expectation(
    pair_matrix: torch.Tensor,
    blocked_orbital: torch.Tensor,
    pairs: int,
    operator: torch.Tensor,
) -> torch.Tensor:
    """Return an unnormalized blocked-AGP one-body expectation."""

    return blocked_agp_one_body_transition(
        pair_matrix,
        blocked_orbital,
        pair_matrix,
        blocked_orbital,
        pairs,
        operator,
    )


def agp_one_body_transition(
    bra_pair_matrix: torch.Tensor,
    ket_pair_matrix: torch.Tensor,
    pairs: int,
    operator: torch.Tensor,
) -> torch.Tensor:
    """Return ``<AGP(F)|sum_i h(i)|AGP(G)>`` polynomially."""

    dimension = _validate_pair_matrix(bra_pair_matrix, pairs)
    _validate_pair_matrix(ket_pair_matrix, pairs)
    if operator.shape != (dimension, dimension):
        raise ValueError("operator must have shape (D, D)")
    direction = operator @ ket_pair_matrix + ket_pair_matrix @ operator.transpose(0, 1)
    _, derivative = _overlap_recurrence(
        bra_pair_matrix, ket_pair_matrix, pairs, ket_direction=direction
    )
    assert derivative is not None
    return derivative


def agp_one_body_expectation(
    pair_matrix: torch.Tensor,
    pairs: int,
    operator: torch.Tensor,
) -> torch.Tensor:
    """Return ``<AGP|sum_i operator(i)|AGP>`` without state materialization."""

    return agp_one_body_transition(pair_matrix, pair_matrix, pairs, operator)


def _complex_derivative(
    value: torch.Tensor, variable: torch.Tensor, *, create_graph: bool
) -> torch.Tensor:
    if not value.is_complex():
        return torch.autograd.grad(
            value,
            variable,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
    real_derivative = torch.autograd.grad(
        value.real,
        variable,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    imaginary_derivative = torch.autograd.grad(
        value.imag,
        variable,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return torch.complex(real_derivative, imaginary_derivative)


def _overlap_mixed_recurrence_batched(
    bra_pair_matrix: torch.Tensor,
    ket_pair_matrix: torch.Tensor,
    pairs: int,
    first_directions: torch.Tensor,
    second_directions: torch.Tensor,
    mixed_directions: torch.Tensor,
) -> torch.Tensor:
    """Return batched mixed overlap derivatives by a second-order recurrence."""

    dimension = bra_pair_matrix.shape[0]
    terms = first_directions.shape[0]
    bra_scale = _detached_max_abs_scale(bra_pair_matrix)
    ket_scale = _detached_max_abs_scale(ket_pair_matrix)
    normalized_bra_adjoint = (
        bra_pair_matrix / bra_scale
    ).conj().transpose(0, 1)
    overlap_matrix = normalized_bra_adjoint @ (ket_pair_matrix / ket_scale)
    first_matrices = torch.matmul(
        normalized_bra_adjoint, first_directions / ket_scale
    )
    second_matrices = torch.matmul(
        normalized_bra_adjoint, second_directions / ket_scale
    )
    mixed_matrices = torch.matmul(
        normalized_bra_adjoint, mixed_directions / ket_scale
    )

    power_matrix = torch.eye(
        dimension, dtype=overlap_matrix.dtype, device=overlap_matrix.device
    )
    zero_batch = torch.zeros(
        (terms, dimension, dimension),
        dtype=overlap_matrix.dtype,
        device=overlap_matrix.device,
    )
    first_power = zero_batch
    second_power = zero_batch
    mixed_power = zero_batch
    zero_scalar = torch.zeros(
        (), dtype=overlap_matrix.dtype, device=overlap_matrix.device
    )
    zero_terms = torch.zeros(
        terms, dtype=overlap_matrix.dtype, device=overlap_matrix.device
    )
    traces = [zero_scalar]
    first_traces = [zero_terms]
    second_traces = [zero_terms]
    mixed_traces = [zero_terms]
    for _ in range(1, pairs + 1):
        next_mixed = (
            torch.matmul(mixed_power, overlap_matrix)
            + torch.matmul(first_power, second_matrices)
            + torch.matmul(second_power, first_matrices)
            + torch.matmul(power_matrix, mixed_matrices)
        )
        next_first = torch.matmul(first_power, overlap_matrix) + torch.matmul(
            power_matrix, first_matrices
        )
        next_second = torch.matmul(second_power, overlap_matrix) + torch.matmul(
            power_matrix, second_matrices
        )
        power_matrix = power_matrix @ overlap_matrix
        first_power = next_first
        second_power = next_second
        mixed_power = next_mixed
        traces.append(torch.trace(power_matrix))
        first_traces.append(torch.diagonal(first_power, dim1=-2, dim2=-1).sum(-1))
        second_traces.append(torch.diagonal(second_power, dim1=-2, dim2=-1).sum(-1))
        mixed_traces.append(torch.diagonal(mixed_power, dim1=-2, dim2=-1).sum(-1))

    coefficients = [torch.ones_like(zero_scalar)]
    first_coefficients = [zero_terms]
    second_coefficients = [zero_terms]
    mixed_coefficients = [zero_terms]
    for degree in range(1, pairs + 1):
        values = []
        first_values = []
        second_values = []
        mixed_values = []
        for power in range(1, degree + 1):
            sign = -1 if (power + 1) % 2 else 1
            factor = 0.5 * sign
            remainder = degree - power
            values.append(factor * traces[power] * coefficients[remainder])
            first_values.append(
                factor
                * (
                    first_traces[power] * coefficients[remainder]
                    + traces[power] * first_coefficients[remainder]
                )
            )
            second_values.append(
                factor
                * (
                    second_traces[power] * coefficients[remainder]
                    + traces[power] * second_coefficients[remainder]
                )
            )
            mixed_values.append(
                factor
                * (
                    mixed_traces[power] * coefficients[remainder]
                    + first_traces[power] * second_coefficients[remainder]
                    + second_traces[power] * first_coefficients[remainder]
                    + traces[power] * mixed_coefficients[remainder]
                )
            )
        coefficients.append(torch.stack(values).sum() / degree)
        first_coefficients.append(torch.stack(first_values).sum(0) / degree)
        second_coefficients.append(torch.stack(second_values).sum(0) / degree)
        mixed_coefficients.append(torch.stack(mixed_values).sum(0) / degree)
    return (bra_scale * ket_scale).pow(pairs) * mixed_coefficients[pairs]


def agp_two_body_transition_factorized(
    bra_pair_matrix: torch.Tensor,
    ket_pair_matrix: torch.Tensor,
    pairs: int,
    left_operators: torch.Tensor,
    right_operators: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a transition element of a symmetrized factorized two-body operator.

    Each term represents ``w/2 * (A tensor B + B tensor A)`` on a particle
    pair. The mixed derivative of the one-particle transformation generates
    ``sum_{i != j} A(i)B(j)``, hence the final factor of one half.
    """

    dimension = _validate_pair_matrix(bra_pair_matrix, pairs)
    _validate_pair_matrix(ket_pair_matrix, pairs)
    if (
        left_operators.ndim != 3
        or right_operators.shape != left_operators.shape
        or left_operators.shape[1:] != (dimension, dimension)
    ):
        raise ValueError("operator factors must have shape (terms, D, D)")
    if left_operators.dtype != bra_pair_matrix.dtype or right_operators.dtype != bra_pair_matrix.dtype:
        raise ValueError("operator factors and pair matrix must share dtype")
    if left_operators.device != bra_pair_matrix.device or right_operators.device != bra_pair_matrix.device:
        raise ValueError("operator factors and pair matrix must share device")
    terms = left_operators.shape[0]
    if weights is None:
        weights = torch.ones(
            terms, dtype=bra_pair_matrix.dtype, device=bra_pair_matrix.device
        )
    if (
        weights.shape != (terms,)
        or weights.dtype != bra_pair_matrix.dtype
        or weights.device != bra_pair_matrix.device
    ):
        raise ValueError("weights must have shape (terms,) and share dtype/device")

    first_directions = (
        torch.matmul(left_operators, ket_pair_matrix)
        + torch.matmul(ket_pair_matrix, left_operators.transpose(1, 2))
    )
    second_directions = (
        torch.matmul(right_operators, ket_pair_matrix)
        + torch.matmul(ket_pair_matrix, right_operators.transpose(1, 2))
    )
    mixed_directions = (
        torch.matmul(torch.matmul(left_operators, ket_pair_matrix), right_operators.transpose(1, 2))
        + torch.matmul(torch.matmul(right_operators, ket_pair_matrix), left_operators.transpose(1, 2))
    )
    mixed_derivatives = _overlap_mixed_recurrence_batched(
        bra_pair_matrix,
        ket_pair_matrix,
        pairs,
        first_directions,
        second_directions,
        mixed_directions,
    )
    return 0.5 * torch.sum(weights * mixed_derivatives)


def agp_two_body_expectation_factorized(
    pair_matrix: torch.Tensor,
    pairs: int,
    left_operators: torch.Tensor,
    right_operators: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a factorized two-body expectation for one AGP state."""

    return agp_two_body_transition_factorized(
        pair_matrix,
        pair_matrix,
        pairs,
        left_operators,
        right_operators,
        weights,
    )


def blocked_agp_two_body_transition_factorized(
    bra_pair_matrix: torch.Tensor,
    bra_blocked_orbital: torch.Tensor,
    ket_pair_matrix: torch.Tensor,
    ket_blocked_orbital: torch.Tensor,
    pairs: int,
    left_operators: torch.Tensor,
    right_operators: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a factorized two-body transition between blocked AGPs."""

    dimension = _validate_blocked_state(
        bra_pair_matrix, bra_blocked_orbital, pairs
    )
    _validate_blocked_state(ket_pair_matrix, ket_blocked_orbital, pairs)
    if (
        left_operators.ndim != 3
        or right_operators.shape != left_operators.shape
        or left_operators.shape[1:] != (dimension, dimension)
    ):
        raise ValueError("operator factors must have shape (terms, D, D)")
    augmented_bra = blocked_augmented_pair_matrix(
        bra_pair_matrix, bra_blocked_orbital, pairs
    )
    augmented_ket = blocked_augmented_pair_matrix(
        ket_pair_matrix, ket_blocked_orbital, pairs
    )
    extended_left = torch.stack(
        [_extend_blocked_operator(operator) for operator in left_operators]
    )
    extended_right = torch.stack(
        [_extend_blocked_operator(operator) for operator in right_operators]
    )
    result = agp_two_body_transition_factorized(
        augmented_bra,
        augmented_ket,
        pairs + 1,
        extended_left,
        extended_right,
        weights,
    )
    if 2 * (pairs + 1) <= dimension:
        result = result - agp_two_body_transition_factorized(
            bra_pair_matrix,
            ket_pair_matrix,
            pairs + 1,
            left_operators,
            right_operators,
            weights,
        )
    return result


def blocked_agp_two_body_expectation_factorized(
    pair_matrix: torch.Tensor,
    blocked_orbital: torch.Tensor,
    pairs: int,
    left_operators: torch.Tensor,
    right_operators: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return an unnormalized blocked-AGP two-body expectation."""

    return blocked_agp_two_body_transition_factorized(
        pair_matrix,
        blocked_orbital,
        pair_matrix,
        blocked_orbital,
        pairs,
        left_operators,
        right_operators,
        weights,
    )


def _validate_agp_sum(
    pair_matrices: torch.Tensor,
    amplitudes: torch.Tensor,
    pairs: int,
) -> int:
    if pair_matrices.ndim != 3 or pair_matrices.shape[1] != pair_matrices.shape[2]:
        raise ValueError("pair_matrices must have shape (terms, D, D)")
    terms = pair_matrices.shape[0]
    if terms < 1 or amplitudes.shape != (terms,):
        raise ValueError("amplitudes must have one entry per AGP term")
    if amplitudes.dtype != pair_matrices.dtype or amplitudes.device != pair_matrices.device:
        raise ValueError("amplitudes and pair matrices must share dtype/device")
    for term in range(terms):
        _validate_pair_matrix(pair_matrices[term], pairs)
    return terms


def agp_overlap_matrix(
    pair_matrices: torch.Tensor,
    pairs: int,
) -> torch.Tensor:
    """Return ``S[a,b]=<Psi(F_a)|Psi(F_b)>`` for a finite AGP basis."""

    terms = _validate_agp_sum(
        pair_matrices,
        torch.ones(
            pair_matrices.shape[0],
            dtype=pair_matrices.dtype,
            device=pair_matrices.device,
        ),
        pairs,
    )
    rows = []
    for bra in range(terms):
        entries = []
        for ket in range(terms):
            entries.append(
                agp_norm_generating(pair_matrices[bra], pairs).to(
                    pair_matrices.dtype
                )
                if bra == ket
                else agp_overlap_generating(
                    pair_matrices[bra], pair_matrices[ket], pairs
                )
            )
        rows.append(torch.stack(entries))
    return torch.stack(rows)


def agp_one_body_transition_matrix(
    pair_matrices: torch.Tensor,
    pairs: int,
    operator: torch.Tensor,
) -> torch.Tensor:
    """Return all finite-AGP one-body transition matrix elements."""

    terms = _validate_agp_sum(
        pair_matrices,
        torch.ones(
            pair_matrices.shape[0],
            dtype=pair_matrices.dtype,
            device=pair_matrices.device,
        ),
        pairs,
    )
    return torch.stack(
        [
            torch.stack(
                [
                    agp_one_body_transition(
                        pair_matrices[bra],
                        pair_matrices[ket],
                        pairs,
                        operator,
                    )
                    for ket in range(terms)
                ]
            )
            for bra in range(terms)
        ]
    )


def agp_two_body_transition_matrix_factorized(
    pair_matrices: torch.Tensor,
    pairs: int,
    left_operators: torch.Tensor,
    right_operators: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return all finite-AGP factorized two-body transition elements."""

    terms = _validate_agp_sum(
        pair_matrices,
        torch.ones(
            pair_matrices.shape[0],
            dtype=pair_matrices.dtype,
            device=pair_matrices.device,
        ),
        pairs,
    )
    return torch.stack(
        [
            torch.stack(
                [
                    agp_two_body_transition_factorized(
                        pair_matrices[bra],
                        pair_matrices[ket],
                        pairs,
                        left_operators,
                        right_operators,
                        weights,
                    )
                    for ket in range(terms)
                ]
            )
            for bra in range(terms)
        ]
    )


def agp_sum_norm(
    pair_matrices: torch.Tensor,
    amplitudes: torch.Tensor,
    pairs: int,
) -> torch.Tensor:
    """Return the norm of a finite AGP sum in ``O(K^2 M D^3)`` time."""

    terms = _validate_agp_sum(pair_matrices, amplitudes, pairs)
    assert terms == pair_matrices.shape[0]
    overlap = agp_overlap_matrix(pair_matrices, pairs)
    return torch.einsum("a,ab,b->", amplitudes.conj(), overlap, amplitudes).real


def agp_sum_one_body_expectation(
    pair_matrices: torch.Tensor,
    amplitudes: torch.Tensor,
    pairs: int,
    operator: torch.Tensor,
) -> torch.Tensor:
    """Return an unnormalized one-body expectation for a finite AGP sum."""

    terms = _validate_agp_sum(pair_matrices, amplitudes, pairs)
    assert terms == pair_matrices.shape[0]
    transition = agp_one_body_transition_matrix(
        pair_matrices, pairs, operator
    )
    return torch.einsum(
        "a,ab,b->", amplitudes.conj(), transition, amplitudes
    )


def agp_sum_two_body_expectation_factorized(
    pair_matrices: torch.Tensor,
    amplitudes: torch.Tensor,
    pairs: int,
    left_operators: torch.Tensor,
    right_operators: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a factorized two-body expectation for a finite AGP sum."""

    terms = _validate_agp_sum(pair_matrices, amplitudes, pairs)
    assert terms == pair_matrices.shape[0]
    transition = agp_two_body_transition_matrix_factorized(
        pair_matrices,
        pairs,
        left_operators,
        right_operators,
        weights,
    )
    return torch.einsum(
        "a,ab,b->", amplitudes.conj(), transition, amplitudes
    )


def agp_structural_counts(
    dimension: int, pairs: int, channels: int
) -> dict[str, int]:
    """Return exact representation counts for a channel-factorized AGP."""

    if pairs < 1 or pairs > channels or 2 * pairs > dimension:
        raise ValueError("require 1 <= pairs <= channels and 2*pairs <= D")
    return {
        "particles": 2 * pairs,
        "pair_matrix_parameters": dimension * (dimension - 1) // 2,
        "femps_internal_bond": channels,
        "nonzero_ordered_paths": math.comb(channels, pairs),
        "unrestricted_path_bound": channels ** (2 * pairs - 1),
        "full_particle_coefficients": dimension ** (2 * pairs),
        "exterior_coefficients": math.comb(dimension, 2 * pairs),
    }
