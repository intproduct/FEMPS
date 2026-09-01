"""Unbounded soft-Coulomb MPO from a Fourier--Bessel separation."""

from __future__ import annotations

import math

import numpy as np
import torch

from femps.basis.odd_hermite import odd_hermite_characteristic_matrices
from femps.baselines.ordered_distance_mpo import sum_mpos


def _distance_characteristic_matrices(
    distance_basis: str,
    basis_order: int,
    frequencies: torch.Tensor,
    distance_scale: float,
    distance_scale_ratio: float,
    *,
    quadrature_order: int | None,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if distance_basis == "odd_hermite":
        return odd_hermite_characteristic_matrices(
            basis_order,
            frequencies,
            distance_scale,
            quadrature_order=quadrature_order,
            dtype=dtype,
            device=device,
        )
    if distance_basis == "multiscale_odd_hermite":
        from femps.basis.multiscale_odd_hermite import (
            multiscale_odd_hermite_characteristic_matrices,
        )

        return multiscale_odd_hermite_characteristic_matrices(
            basis_order,
            frequencies,
            distance_scale,
            distance_scale_ratio,
            quadrature_order=quadrature_order,
            dtype=dtype,
            device=device,
        )
    raise ValueError(
        "Fourier interaction requires odd_hermite or "
        "multiscale_odd_hermite"
    )


def soft_coulomb_fourier_rule(
    order: int,
    *,
    softening: float = 1.0,
    dimensionless_cutoff: float = 30.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return nodes and coefficients for the soft-Coulomb cosine integral.

    The exact transform pair is

    ``1/sqrt(s^2+a^2) = (2/pi) int_0^inf K0(a k) cos(k s) dk``.

    The finite rule truncates at ``a*k=dimensionless_cutoff`` and maps
    ``k=k_max*u^2`` before Gauss--Legendre quadrature.  The square map removes
    the integrable logarithmic endpoint singularity of ``K0``.  Both the
    cutoff and quadrature order remain explicit numerical controls.
    """

    if order < 1 or softening <= 0 or dimensionless_cutoff <= 0:
        raise ValueError("order, softening, and cutoff must be positive")
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(order)
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    unit_nodes = torch.as_tensor(
        (raw_nodes + 1) / 2, dtype=real_dtype, device=device
    )
    raw_weights_tensor = torch.as_tensor(
        raw_weights, dtype=real_dtype, device=device
    )
    maximum_frequency = dimensionless_cutoff / softening
    frequencies = maximum_frequency * unit_nodes.square()
    # du=raw_weight/2 and dk/du=2*k_max*u.
    frequency_weights = raw_weights_tensor * maximum_frequency * unit_nodes
    coefficients = (
        2
        / math.pi
        * frequency_weights
        * torch.special.modified_bessel_k0(softening * frequencies)
    )
    return frequencies.to(dtype), coefficients.to(dtype)


def soft_coulomb_fourier_sampled_error(
    maximum_separation: float,
    order: int,
    *,
    softening: float = 1.0,
    dimensionless_cutoff: float = 30.0,
    samples: int = 20001,
) -> float:
    """Return a finite-interval sampled diagnostic for the Fourier rule."""

    if maximum_separation < 0 or samples < 2:
        raise ValueError("separation must be nonnegative and samples at least two")
    frequencies, coefficients = soft_coulomb_fourier_rule(
        order,
        softening=softening,
        dimensionless_cutoff=dimensionless_cutoff,
    )
    separation = torch.linspace(
        0,
        maximum_separation,
        samples,
        dtype=frequencies.real.dtype,
        device=frequencies.device,
    )
    observed = torch.einsum(
        "k,ks->s", coefficients, torch.cos(frequencies[:, None] * separation)
    )
    expected = torch.rsqrt(separation.square() + softening**2)
    return float(torch.max(torch.abs(observed - expected)))


def ordered_continuous_fourier_soft_coulomb_pair_mpo(
    particles: int,
    basis_order: int,
    distance_scale: float,
    left_particle: int,
    right_particle: int,
    fourier_order: int,
    *,
    distance_basis: str = "odd_hermite",
    distance_scale_ratio: float = 2.0,
    softening: float = 1.0,
    dimensionless_cutoff: float = 30.0,
    local_quadrature_order: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Return one pair interaction as a real two-channel rotation MPO.

    For each Fourier node, projected cosine/sine multiplication matrices form
    a rotation block.  Multiplying the blocks across consecutive positive gaps
    yields ``cos(k * sum(r_i))`` with bond two per node.
    """

    if (
        particles < 2
        or basis_order < 1
        or distance_scale <= 0
        or not 0 <= left_particle < right_particle < particles
        or fourier_order < 1
        or softening <= 0
    ):
        raise ValueError("invalid particles, basis, pair, Fourier order, or scale")
    try:
        from latticetn.mpo import MPO
    except ImportError as exc:
        raise ImportError("latticeTN is required for Fourier interaction MPOs") from exc

    frequencies, coefficients = soft_coulomb_fourier_rule(
        fourier_order,
        softening=softening,
        dimensionless_cutoff=dimensionless_cutoff,
        dtype=dtype,
        device=device,
    )
    cosine, sine = _distance_characteristic_matrices(
        distance_basis,
        basis_order,
        frequencies,
        distance_scale,
        distance_scale_ratio,
        quadrature_order=local_quadrature_order,
        dtype=dtype,
        device=device,
    )
    identity = torch.eye(basis_order, dtype=dtype, device=device)
    start = left_particle + 1
    end = right_particle
    tensors = []
    for site in range(particles):
        if site < start or site > end:
            tensors.append(
                identity.transpose(0, 1).reshape(
                    1, 1, basis_order, basis_order
                )
            )
            continue
        if start == end:
            operator = torch.einsum("k,kij->ij", coefficients, cosine)
            tensors.append(
                operator.transpose(0, 1).reshape(
                    1, 1, basis_order, basis_order
                )
            )
            continue
        bond = 2 * fourier_order
        if site == start:
            tensor = torch.zeros(
                1, bond, basis_order, basis_order, dtype=dtype, device=device
            )
            for node in range(fourier_order):
                tensor[0, 2 * node] = (
                    coefficients[node] * cosine[node].transpose(0, 1)
                )
                tensor[0, 2 * node + 1] = (
                    -coefficients[node] * sine[node].transpose(0, 1)
                )
            tensors.append(tensor)
            continue
        if site == end:
            tensor = torch.zeros(
                bond, 1, basis_order, basis_order, dtype=dtype, device=device
            )
            for node in range(fourier_order):
                tensor[2 * node, 0] = cosine[node].transpose(0, 1)
                tensor[2 * node + 1, 0] = sine[node].transpose(0, 1)
            tensors.append(tensor)
            continue
        tensor = torch.zeros(
            bond, bond, basis_order, basis_order, dtype=dtype, device=device
        )
        for node in range(fourier_order):
            cosine_operator = cosine[node].transpose(0, 1)
            sine_operator = sine[node].transpose(0, 1)
            tensor[2 * node, 2 * node] = cosine_operator
            tensor[2 * node, 2 * node + 1] = -sine_operator
            tensor[2 * node + 1, 2 * node] = sine_operator
            tensor[2 * node + 1, 2 * node + 1] = cosine_operator
        tensors.append(tensor)
    return MPO(
        tensors,
        length=particles,
        dim=basis_order,
        dtype=dtype,
        device=device,
    )


def ordered_continuous_fourier_soft_coulomb_mpo(
    particles: int,
    basis_order: int,
    distance_scale: float,
    fourier_order: int,
    *,
    construction: str = "compact",
    **kwargs,
):
    """Return all pair interactions in compact or direct-pair form.

    The compact construction propagates four real states per Fourier node.
    If ``c`` and ``s`` contain the cosine and sine sums from the particles to
    the left of the current one, crossing the next positive gap updates them
    by a rotation and accumulates the new ``c`` in the total ``T``.  This
    represents every particle pair with bond ``4 * fourier_order``, independent
    of particle count.  ``direct_pairs`` is retained as an audit construction.
    """

    if construction == "compact":
        return ordered_continuous_fourier_soft_coulomb_compact_mpo(
            particles,
            basis_order,
            distance_scale,
            fourier_order,
            **kwargs,
        )
    if construction != "direct_pairs":
        raise ValueError("construction must be 'compact' or 'direct_pairs'")

    return sum_mpos(
        [
            ordered_continuous_fourier_soft_coulomb_pair_mpo(
                particles,
                basis_order,
                distance_scale,
                left,
                right,
                fourier_order,
                **kwargs,
            )
            for left in range(particles)
            for right in range(left + 1, particles)
        ]
    )


def ordered_continuous_fourier_soft_coulomb_compact_mpo(
    particles: int,
    basis_order: int,
    distance_scale: float,
    fourier_order: int,
    *,
    distance_basis: str = "odd_hermite",
    distance_scale_ratio: float = 2.0,
    softening: float = 1.0,
    dimensionless_cutoff: float = 30.0,
    local_quadrature_order: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Return the all-pair Fourier interaction with four states per node."""

    if (
        particles < 2
        or basis_order < 1
        or distance_scale <= 0
        or fourier_order < 1
        or softening <= 0
    ):
        raise ValueError("invalid particles, basis, Fourier order, or scale")
    if particles == 2:
        return ordered_continuous_fourier_soft_coulomb_pair_mpo(
            particles,
            basis_order,
            distance_scale,
            0,
            1,
            fourier_order,
            distance_basis=distance_basis,
            distance_scale_ratio=distance_scale_ratio,
            softening=softening,
            dimensionless_cutoff=dimensionless_cutoff,
            local_quadrature_order=local_quadrature_order,
            dtype=dtype,
            device=device,
        )
    try:
        from latticetn.mpo import MPO
    except ImportError as exc:
        raise ImportError("latticeTN is required for Fourier interaction MPOs") from exc

    frequencies, coefficients = soft_coulomb_fourier_rule(
        fourier_order,
        softening=softening,
        dimensionless_cutoff=dimensionless_cutoff,
        dtype=dtype,
        device=device,
    )
    cosine, sine = _distance_characteristic_matrices(
        distance_basis,
        basis_order,
        frequencies,
        distance_scale,
        distance_scale_ratio,
        quadrature_order=local_quadrature_order,
        dtype=dtype,
        device=device,
    )
    identity = torch.eye(basis_order, dtype=dtype, device=device).transpose(0, 1)
    bond = 4 * fourier_order
    first = torch.zeros(
        1, bond, basis_order, basis_order, dtype=dtype, device=device
    )
    for node in range(fourier_order):
        first[0, 4 * node] = coefficients[node] * identity
    tensors = [first]

    for _site in range(1, particles - 1):
        bulk = torch.zeros(
            bond, bond, basis_order, basis_order, dtype=dtype, device=device
        )
        for node in range(fourier_order):
            base = 4 * node
            cosine_operator = cosine[node].transpose(0, 1)
            sine_operator = sine[node].transpose(0, 1)
            # A row state [1, c, s, T] crosses one positive gap.  The new
            # c/s pair is a rotation of [c + 1, s], and T accumulates new c.
            bulk[base, base] = identity
            bulk[base, base + 1] = cosine_operator
            bulk[base + 1, base + 1] = cosine_operator
            bulk[base + 2, base + 1] = -sine_operator
            bulk[base, base + 2] = sine_operator
            bulk[base + 1, base + 2] = sine_operator
            bulk[base + 2, base + 2] = cosine_operator
            bulk[base, base + 3] = cosine_operator
            bulk[base + 1, base + 3] = cosine_operator
            bulk[base + 2, base + 3] = -sine_operator
            bulk[base + 3, base + 3] = identity
        tensors.append(bulk)

    # The final site applies one more recurrence step and selects its total T.
    last = torch.zeros(
        bond, 1, basis_order, basis_order, dtype=dtype, device=device
    )
    for node in range(fourier_order):
        base = 4 * node
        last[base, 0] = cosine[node].transpose(0, 1)
        last[base + 1, 0] = cosine[node].transpose(0, 1)
        last[base + 2, 0] = -sine[node].transpose(0, 1)
        last[base + 3, 0] = identity
    tensors.append(last)
    return MPO(
        tensors,
        length=particles,
        dim=basis_order,
        dtype=dtype,
        device=device,
    )


def ordered_continuous_fourier_hamiltonian_mpo(
    particles: int,
    basis_order: int,
    distance_scale: float,
    fourier_order: int,
    *,
    distance_basis: str = "odd_hermite",
    distance_scale_ratio: float = 2.0,
    center_of_mass_length: float | None = None,
    omega: float = 1.0,
    coupling: float = 1.0,
    softening: float = 1.0,
    dimensionless_cutoff: float = 30.0,
    local_quadrature_order: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Return the odd-Hermite noninteracting plus unbounded interaction MPO."""

    if coupling < 0:
        raise ValueError("coupling must be nonnegative")
    from femps.baselines.ordered_continuous_mpo import (
        ordered_continuous_noninteracting_mpo,
    )

    noninteracting = ordered_continuous_noninteracting_mpo(
        particles,
        basis_order,
        distance_scale,
        distance_basis=distance_basis,
        distance_scale_ratio=distance_scale_ratio,
        center_of_mass_length=center_of_mass_length,
        omega=omega,
        dtype=dtype,
        device=device,
    )
    if coupling == 0:
        return noninteracting
    interaction = ordered_continuous_fourier_soft_coulomb_mpo(
        particles,
        basis_order,
        distance_scale,
        fourier_order,
        distance_basis=distance_basis,
        distance_scale_ratio=distance_scale_ratio,
        softening=softening,
        dimensionless_cutoff=dimensionless_cutoff,
        local_quadrature_order=local_quadrature_order,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        interaction.tensors[0].mul_(coupling)
    return sum_mpos([noninteracting, interaction])


def _contract_fourier_bulk_from_left(
    transfer: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    identity: torch.Tensor,
) -> torch.Tensor:
    """Apply all sparse four-state recurrence blocks to a left transfer."""

    retained = transfer.shape[0]
    fourier_order = cosine.shape[0]
    states = transfer.reshape(retained, fourier_order, 4)
    constant, cosine_sum, sine_sum, total = states.unbind(dim=2)
    common = constant + cosine_sum
    cosine_update = (
        common[..., None, None] * cosine[None]
        - sine_sum[..., None, None] * sine[None]
    )
    sine_update = (
        common[..., None, None] * sine[None]
        + sine_sum[..., None, None] * cosine[None]
    )
    output = torch.empty(
        retained,
        fourier_order,
        4,
        identity.shape[0],
        identity.shape[1],
        dtype=transfer.dtype,
        device=transfer.device,
    )
    output[:, :, 0] = constant[..., None, None] * identity
    output[:, :, 1] = cosine_update
    output[:, :, 2] = sine_update
    output[:, :, 3] = cosine_update + total[..., None, None] * identity
    return output.reshape(retained, 4 * fourier_order, *identity.shape)


def _contract_fourier_last_from_left(
    transfer: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    identity: torch.Tensor,
) -> torch.Tensor:
    """Apply the final recurrence step and sum all Fourier-node totals."""

    retained = transfer.shape[0]
    states = transfer.reshape(retained, cosine.shape[0], 4)
    constant, cosine_sum, sine_sum, total = states.unbind(dim=2)
    value = (
        (constant + cosine_sum)[..., None, None] * cosine[None]
        - sine_sum[..., None, None] * sine[None]
        + total[..., None, None] * identity
    )
    return value.sum(dim=1)


def ordered_continuous_fourier_hamiltonian_compressed_mpo(
    particles: int,
    basis_order: int,
    distance_scale: float,
    fourier_order: int,
    maximum_bond: int,
    *,
    distance_basis: str = "odd_hermite",
    distance_scale_ratio: float = 2.0,
    relative_tolerance: float = 0.0,
    center_of_mass_length: float | None = None,
    omega: float = 1.0,
    coupling: float = 1.0,
    softening: float = 1.0,
    dimensionless_cutoff: float = 30.0,
    local_quadrature_order: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Build the compressed Hamiltonian without dense raw Fourier bulk blocks.

    This performs the same left-to-right Hilbert--Schmidt SVD as
    :func:`compress_mpo`. At each bulk site, the incoming transfer is applied
    directly to the sparse four-state Fourier recurrence. The theoretical raw
    direct-sum shape is recorded, but no ``(4M)^2 D^2`` tensor is materialized.
    """

    if maximum_bond < 1 or relative_tolerance < 0:
        raise ValueError("maximum_bond must be positive and tolerance nonnegative")
    if coupling < 0:
        raise ValueError("coupling must be nonnegative")
    from femps.baselines.ordered_continuous_mpo import (
        ordered_continuous_noninteracting_mpo,
    )
    from femps.baselines.ordered_distance_mpo import (
        _compress_left_mpo_tensor,
        compress_mpo,
    )

    noninteracting = ordered_continuous_noninteracting_mpo(
        particles,
        basis_order,
        distance_scale,
        distance_basis=distance_basis,
        distance_scale_ratio=distance_scale_ratio,
        center_of_mass_length=center_of_mass_length,
        omega=omega,
        dtype=dtype,
        device=device,
    )
    if coupling == 0 or particles == 2:
        raw = (
            noninteracting
            if coupling == 0
            else ordered_continuous_fourier_hamiltonian_mpo(
                particles,
                basis_order,
                distance_scale,
                fourier_order,
                distance_basis=distance_basis,
                distance_scale_ratio=distance_scale_ratio,
                center_of_mass_length=center_of_mass_length,
                omega=omega,
                coupling=coupling,
                softening=softening,
                dimensionless_cutoff=dimensionless_cutoff,
                local_quadrature_order=local_quadrature_order,
                dtype=dtype,
                device=device,
            )
        )
        compressed, ranks, discarded = compress_mpo(
            raw,
            maximum_bond,
            relative_tolerance=relative_tolerance,
        )
        diagnostics = {
            "construction": "small_system_dense_free_fallback",
            "dense_raw_fourier_bulk_materialized": False,
            "theoretical_raw_maximum_bond": max(
                max(tensor.shape[:2]) for tensor in raw.tensors
            ),
            "theoretical_raw_tensor_elements": sum(
                tensor.numel() for tensor in raw.tensors
            ),
            "maximum_intermediate_tensor_elements": max(
                tensor.numel() for tensor in raw.tensors
            ),
            "retained_ranks": tuple(ranks),
            "local_discarded_norm_not_global_certificate": discarded,
        }
        return compressed, diagnostics

    try:
        from latticetn.mpo import MPO
    except ImportError as exc:
        raise ImportError("latticeTN is required for Fourier interaction MPOs") from exc

    frequencies, coefficients = soft_coulomb_fourier_rule(
        fourier_order,
        softening=softening,
        dimensionless_cutoff=dimensionless_cutoff,
        dtype=dtype,
        device=device,
    )
    cosine, sine = _distance_characteristic_matrices(
        distance_basis,
        basis_order,
        frequencies,
        distance_scale,
        distance_scale_ratio,
        quadrature_order=local_quadrature_order,
        dtype=dtype,
        device=device,
    )
    cosine = cosine.transpose(-2, -1)
    sine = sine.transpose(-2, -1)
    identity = torch.eye(
        basis_order, dtype=dtype, device=device
    ).transpose(0, 1)
    fourier_bond = 4 * fourier_order

    first_fourier = torch.zeros(
        1,
        fourier_bond,
        basis_order,
        basis_order,
        dtype=dtype,
        device=device,
    )
    first_fourier[0, 0::4] = (
        coupling * coefficients[:, None, None] * identity
    )
    first = torch.cat([noninteracting.tensors[0], first_fourier], dim=1)
    compressed_first, transfer, retained, discarded_squared = (
        _compress_left_mpo_tensor(
            first, maximum_bond, relative_tolerance
        )
    )
    tensors = [compressed_first]
    ranks = [retained]
    maximum_intermediate = first.numel()

    theoretical_raw_elements = first.numel()
    theoretical_raw_maximum_bond = first.shape[1]
    for site in range(1, particles - 1):
        noninteracting_site = noninteracting.tensors[site]
        noninteracting_left = noninteracting_site.shape[0]
        if transfer.shape[1] != noninteracting_left + fourier_bond:
            raise RuntimeError("structured Fourier and noninteracting bonds disagree")
        noninteracting_contracted = torch.einsum(
            "ar,rsij->asij",
            transfer[:, :noninteracting_left],
            noninteracting_site,
        )
        fourier_contracted = _contract_fourier_bulk_from_left(
            transfer[:, noninteracting_left:], cosine, sine, identity
        )
        contracted = torch.cat(
            [noninteracting_contracted, fourier_contracted], dim=1
        )
        maximum_intermediate = max(maximum_intermediate, contracted.numel())
        theoretical_left = noninteracting_site.shape[0] + fourier_bond
        theoretical_right = noninteracting_site.shape[1] + fourier_bond
        theoretical_raw_elements += (
            theoretical_left
            * theoretical_right
            * basis_order
            * basis_order
        )
        theoretical_raw_maximum_bond = max(
            theoretical_raw_maximum_bond,
            theoretical_left,
            theoretical_right,
        )
        compressed_site, transfer, retained, site_discarded_squared = (
            _compress_left_mpo_tensor(
                contracted, maximum_bond, relative_tolerance
            )
        )
        tensors.append(compressed_site)
        ranks.append(retained)
        discarded_squared = discarded_squared + site_discarded_squared

    noninteracting_last = noninteracting.tensors[-1]
    noninteracting_left = noninteracting_last.shape[0]
    if transfer.shape[1] != noninteracting_left + fourier_bond:
        raise RuntimeError("structured Fourier final bond disagrees")
    last = torch.einsum(
        "ar,rsij->asij",
        transfer[:, :noninteracting_left],
        noninteracting_last,
    )
    last = last + _contract_fourier_last_from_left(
        transfer[:, noninteracting_left:], cosine, sine, identity
    )[:, None]
    tensors.append(last)
    maximum_intermediate = max(maximum_intermediate, last.numel())
    theoretical_last_left = noninteracting_last.shape[0] + fourier_bond
    theoretical_raw_elements += (
        theoretical_last_left * basis_order * basis_order
    )
    theoretical_raw_maximum_bond = max(
        theoretical_raw_maximum_bond, theoretical_last_left
    )
    compressed = MPO(
        tensors,
        length=particles,
        dim=basis_order,
        dtype=dtype,
        device=device,
    )
    diagnostics = {
        "construction": "incremental_structured_left_svd",
        "dense_raw_fourier_bulk_materialized": False,
        "theoretical_raw_maximum_bond": theoretical_raw_maximum_bond,
        "theoretical_raw_tensor_elements": theoretical_raw_elements,
        "maximum_intermediate_tensor_elements": maximum_intermediate,
        "retained_ranks": tuple(ranks),
        "local_discarded_norm_not_global_certificate": torch.sqrt(
            discarded_squared
        ),
    }
    return compressed, diagnostics
