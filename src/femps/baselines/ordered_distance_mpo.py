"""Native latticeTN MPS/MPO objects for finite-box ordered gap coordinates."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from femps.baselines.ordered_functional_mps import particle_tensor_to_mps_tensors
from femps.ordered_distance import ordered_values_to_gap_tensor


def _latticetn_classes():
    try:
        from latticetn.mpo import MPO
        from latticetn.mps import MPS
    except ImportError as exc:
        raise ImportError(
            "latticeTN is required; install the sibling repository with "
            "`python -m pip install -e ../latticeTN`"
        ) from exc
    return MPS, MPO


def local_gap_operators(
    local_dimension: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Return identity, number, square, and unit-amplitude gap shifts."""

    if local_dimension < 1:
        raise ValueError("local_dimension must be positive")
    identity = torch.eye(local_dimension, dtype=dtype, device=device)
    values = torch.arange(local_dimension, dtype=torch.float64, device=device).to(dtype)
    number = torch.diag(values)
    raise_gap = torch.zeros_like(identity)
    if local_dimension > 1:
        indices = torch.arange(local_dimension - 1, device=device)
        raise_gap[indices + 1, indices] = 1
    return {
        "identity": identity,
        "number": number,
        "number_squared": number @ number,
        "raise": raise_gap,
        "lower": raise_gap.mH,
    }


def product_sum_mpo(
    product_terms: Sequence[tuple[torch.Tensor, ...]],
):
    """Build an MPO equal to a sum of explicit local-operator products.

    Each term supplies one conventional ``(out,in)`` local matrix per site.
    The direct-sum construction has bond equal to the number of terms.
    """

    if not product_terms:
        raise ValueError("at least one product term is required")
    sites = len(product_terms[0])
    if sites < 1 or any(len(term) != sites for term in product_terms):
        raise ValueError("all product terms must have the same positive length")
    dimension = product_terms[0][0].shape[0]
    dtype = product_terms[0][0].dtype
    device = product_terms[0][0].device
    if any(
        operator.shape != (dimension, dimension)
        or operator.dtype != dtype
        or operator.device != device
        for term in product_terms
        for operator in term
    ):
        raise ValueError("all local operators must share shape, dtype, and device")
    _, MPO = _latticetn_classes()
    if sites == 1:
        summed = torch.stack([term[0] for term in product_terms]).sum(dim=0)
        return MPO(
            [summed.transpose(0, 1).reshape(1, 1, dimension, dimension)],
            length=1,
            dim=dimension,
            dtype=dtype,
            device=device,
        )
    terms = len(product_terms)
    tensors = []
    first = torch.zeros(1, terms, dimension, dimension, dtype=dtype, device=device)
    for channel, term in enumerate(product_terms):
        first[0, channel] = term[0].transpose(0, 1)
    tensors.append(first)
    for site in range(1, sites - 1):
        bulk = torch.zeros(
            terms, terms, dimension, dimension, dtype=dtype, device=device
        )
        for channel, term in enumerate(product_terms):
            bulk[channel, channel] = term[site].transpose(0, 1)
        tensors.append(bulk)
    last = torch.zeros(terms, 1, dimension, dimension, dtype=dtype, device=device)
    for channel, term in enumerate(product_terms):
        last[channel, 0] = term[-1].transpose(0, 1)
    tensors.append(last)
    return MPO(
        tensors,
        length=sites,
        dim=dimension,
        dtype=dtype,
        device=device,
    )


def sum_mpos(mpos: Sequence):
    """Return the block-direct-sum MPO representing a sum of MPOs."""

    if not mpos:
        raise ValueError("at least one MPO is required")
    reference = mpos[0]
    if any(
        mpo.length != reference.length
        or mpo.dim != reference.dim
        or mpo.dtype != reference.dtype
        or mpo.tensors[0].device != reference.tensors[0].device
        for mpo in mpos
    ):
        raise ValueError("MPOs must share length, dimension, dtype, and device")
    _, MPO = _latticetn_classes()
    if reference.length == 1:
        tensor = torch.stack([mpo.tensors[0] for mpo in mpos]).sum(dim=0)
        return MPO(
            [tensor],
            length=1,
            dim=reference.dim,
            dtype=reference.dtype,
            device=reference.tensors[0].device,
        )
    tensors = []
    for site in range(reference.length):
        if site == 0:
            tensors.append(torch.cat([mpo.tensors[site] for mpo in mpos], dim=1))
            continue
        if site == reference.length - 1:
            tensors.append(torch.cat([mpo.tensors[site] for mpo in mpos], dim=0))
            continue
        left_total = sum(mpo.tensors[site].shape[0] for mpo in mpos)
        right_total = sum(mpo.tensors[site].shape[1] for mpo in mpos)
        tensor = torch.zeros(
            left_total,
            right_total,
            reference.dim,
            reference.dim,
            dtype=reference.dtype,
            device=reference.device,
        )
        left_offset = 0
        right_offset = 0
        for mpo in mpos:
            block = mpo.tensors[site]
            tensor[
                left_offset : left_offset + block.shape[0],
                right_offset : right_offset + block.shape[1],
            ] = block
            left_offset += block.shape[0]
            right_offset += block.shape[1]
        tensors.append(tensor)
    return MPO(
        tensors,
        length=reference.length,
        dim=reference.dim,
        dtype=reference.dtype,
        device=reference.tensors[0].device,
    )


def _compress_left_mpo_tensor(
    tensor: torch.Tensor,
    max_bond: int,
    relative_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Factor one MPO site and return its right-going transfer matrix."""

    left, right, physical_in, physical_out = tensor.shape
    matrix = tensor.permute(0, 2, 3, 1).reshape(
        left * physical_in * physical_out, right
    )
    singular_vectors, singular_values, right_vectors = torch.linalg.svd(
        matrix, full_matrices=False
    )
    retained = min(max_bond, singular_values.numel())
    if relative_tolerance > 0 and singular_values.numel() and singular_values[0] > 0:
        retained = min(
            retained,
            max(
                1,
                int(
                    torch.count_nonzero(
                        singular_values
                        > relative_tolerance * singular_values[0]
                    )
                ),
            ),
        )
    compressed = singular_vectors[:, :retained].reshape(
        left, physical_in, physical_out, retained
    ).permute(0, 3, 1, 2)
    transfer = singular_values[:retained, None] * right_vectors[:retained]
    discarded_squared = torch.sum(singular_values[retained:] ** 2)
    return compressed, transfer, retained, discarded_squared


def compress_mpo(
    mpo,
    max_bond: int,
    *,
    relative_tolerance: float = 0.0,
):
    """Left-to-right Hilbert-Schmidt SVD compression of an MPO.

    The returned discarded singular-value norm is a local diagnostic, not a
    certified global operator-error bound. Small-system callers must compare
    the represented operator independently.
    """

    if max_bond < 1 or relative_tolerance < 0:
        raise ValueError("max_bond must be positive and tolerance nonnegative")
    _, MPO = _latticetn_classes()
    tensors = [tensor.clone() for tensor in mpo.tensors]
    ranks = []
    discarded_squared = torch.zeros(
        (), dtype=tensors[0].real.dtype, device=tensors[0].device
    )
    for site in range(mpo.length - 1):
        compressed_site, transfer, retained, site_discarded_squared = (
            _compress_left_mpo_tensor(
                tensors[site], max_bond, relative_tolerance
            )
        )
        discarded_squared = discarded_squared + site_discarded_squared
        tensors[site] = compressed_site
        tensors[site + 1] = torch.einsum(
            "ar,rsij->asij", transfer, tensors[site + 1]
        )
        ranks.append(retained)
    compressed = MPO(
        tensors,
        length=mpo.length,
        dim=mpo.dim,
        dtype=mpo.dtype,
        device=tensors[0].device,
    )
    return compressed, tuple(ranks), torch.sqrt(discarded_squared)


def gap_kinetic_harmonic_terms(
    grid_points: int,
    particles: int,
    spacing: float,
    *,
    gap_cutoff: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> list[tuple[torch.Tensor, ...]]:
    """Return product terms for exact finite-difference kinetic plus HO trap."""

    if particles < 1 or grid_points < particles or spacing <= 0:
        raise ValueError("require L >= N >= 1 and positive spacing")
    sites = particles + 1
    holes = grid_points - particles
    cutoff = holes if gap_cutoff is None else gap_cutoff
    if not 0 <= cutoff <= holes:
        raise ValueError("gap_cutoff must satisfy 0 <= gap_cutoff <= L-N")
    local_dimension = cutoff + 1
    operators = local_gap_operators(
        local_dimension, dtype=dtype, device=device
    )
    identity = operators["identity"]
    number = operators["number"]
    terms: list[tuple[torch.Tensor, ...]] = []

    def local_product(assignments: dict[int, torch.Tensor], coefficient: float):
        local = [identity for _ in range(sites)]
        for site, operator in assignments.items():
            local[site] = operator
        local[0] = coefficient * local[0]
        terms.append(tuple(local))

    kinetic_constant = particles / spacing**2
    center = (grid_points - 1) / 2
    offsets = [particle - center for particle in range(particles)]
    trap_constant = 0.5 * spacing**2 * sum(offset**2 for offset in offsets)
    local_product({}, kinetic_constant + trap_constant)

    hopping = -0.5 / spacing**2
    for particle in range(particles):
        local_product(
            {particle: operators["raise"], particle + 1: operators["lower"]},
            hopping,
        )
        local_product(
            {particle: operators["lower"], particle + 1: operators["raise"]},
            hopping,
        )

    for gap_site in range(particles):
        count = particles - gap_site
        linear = spacing**2 * sum(offsets[gap_site:])
        square = 0.5 * spacing**2 * count
        local_product(
            {
                gap_site: linear * number
                + square * operators["number_squared"]
            },
            1.0,
        )
    for left in range(particles):
        for right in range(left + 1, particles):
            coefficient = spacing**2 * (particles - right)
            local_product({left: number, right: number}, coefficient)
    return terms


def gap_kinetic_harmonic_mpo(
    grid_points: int,
    particles: int,
    spacing: float,
    *,
    gap_cutoff: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Return an exact polynomial-bond gap kinetic-plus-harmonic MPO."""

    return product_sum_mpo(
        gap_kinetic_harmonic_terms(
            grid_points,
            particles,
            spacing,
            gap_cutoff=gap_cutoff,
            dtype=dtype,
            device=device,
        )
    )


def gap_charge_projector_mpo(
    grid_points: int,
    particles: int,
    *,
    gap_cutoff: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Project exactly onto ``sum gaps = L-N`` with bond ``L-N+1``."""

    if particles < 1 or grid_points < particles:
        raise ValueError("require L >= N >= 1")
    _, MPO = _latticetn_classes()
    total = grid_points - particles
    cutoff = total if gap_cutoff is None else gap_cutoff
    if not 0 <= cutoff <= total:
        raise ValueError("gap_cutoff must satisfy 0 <= gap_cutoff <= L-N")
    local_dimension = cutoff + 1
    sites = particles + 1
    bond = total + 1
    tensors = []
    for site in range(sites):
        left_states = 1 if site == 0 else bond
        right_states = 1 if site == sites - 1 else bond
        tensor = torch.zeros(
            left_states,
            right_states,
            local_dimension,
            local_dimension,
            dtype=dtype,
            device=device,
        )
        for left_local in range(left_states):
            accumulated = 0 if site == 0 else left_local
            for occupation in range(local_dimension):
                updated = accumulated + occupation
                if site == sites - 1:
                    if updated == total:
                        tensor[left_local, 0, occupation, occupation] = 1
                elif updated <= total:
                    tensor[left_local, updated, occupation, occupation] = 1
        tensors.append(tensor)
    return MPO(
        tensors,
        length=sites,
        dim=local_dimension,
        dtype=dtype,
        device=device,
    )


def gap_soft_coulomb_pair_mpo(
    grid_points: int,
    particles: int,
    spacing: float,
    left_particle: int,
    right_particle: int,
    *,
    gap_cutoff: int | None = None,
    softening: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Return one exact interval-sum pair-potential MPO on the charge sector.

    For particles ``i<j``, their grid separation is
    ``j-i + sum(gaps[i+1:j+1])``.  The counting bond has size ``L-N+1``.
    Paths whose interval occupation exceeds the total physical hole charge
    are omitted, so this MPO is authoritative on ``sum gaps=L-N``.
    """

    if (
        particles < 2
        or grid_points < particles
        or spacing <= 0
        or softening <= 0
        or not 0 <= left_particle < right_particle < particles
    ):
        raise ValueError("invalid grid, particle pair, spacing, or softening")
    _, MPO = _latticetn_classes()
    holes = grid_points - particles
    cutoff = holes if gap_cutoff is None else gap_cutoff
    if not 0 <= cutoff <= holes:
        raise ValueError("gap_cutoff must satisfy 0 <= gap_cutoff <= L-N")
    local_dimension = cutoff + 1
    sites = particles + 1
    start = left_particle + 1
    end = right_particle
    identity = torch.eye(local_dimension, dtype=dtype, device=device)
    tensors = []
    for site in range(sites):
        if site < start or site > end:
            tensors.append(identity.transpose(0, 1).reshape(1, 1, local_dimension, local_dimension))
            continue
        if start == end:
            values = [
                1
                / math.sqrt(
                    (spacing * (right_particle - left_particle + occupation)) ** 2
                    + softening**2
                )
                for occupation in range(local_dimension)
            ]
            diagonal = torch.diag(torch.tensor(values, dtype=dtype, device=device))
            tensors.append(diagonal.transpose(0, 1).reshape(1, 1, local_dimension, local_dimension))
            continue
        if site == start:
            tensor = torch.zeros(
                1,
                holes + 1,
                local_dimension,
                local_dimension,
                dtype=dtype,
                device=device,
            )
            for occupation in range(local_dimension):
                tensor[0, occupation, occupation, occupation] = 1
            tensors.append(tensor)
            continue
        if site == end:
            tensor = torch.zeros(
                holes + 1,
                1,
                local_dimension,
                local_dimension,
                dtype=dtype,
                device=device,
            )
            for accumulated in range(holes + 1):
                for occupation in range(local_dimension):
                    total = accumulated + occupation
                    if total <= holes:
                        separation = right_particle - left_particle + total
                        tensor[accumulated, 0, occupation, occupation] = 1 / math.sqrt(
                            (spacing * separation) ** 2 + softening**2
                        )
            tensors.append(tensor)
            continue
        tensor = torch.zeros(
            holes + 1,
            holes + 1,
            local_dimension,
            local_dimension,
            dtype=dtype,
            device=device,
        )
        for accumulated in range(holes + 1):
            for occupation in range(local_dimension):
                updated = accumulated + occupation
                if updated <= holes:
                    tensor[accumulated, updated, occupation, occupation] = 1
        tensors.append(tensor)
    return MPO(
        tensors,
        length=sites,
        dim=local_dimension,
        dtype=dtype,
        device=device,
    )


def gap_soft_coulomb_mpo(
    grid_points: int,
    particles: int,
    spacing: float,
    *,
    gap_cutoff: int | None = None,
    softening: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Return the exact finite-grid sum of all interval-count pair MPOs."""

    pair_mpos = [
        gap_soft_coulomb_pair_mpo(
            grid_points,
            particles,
            spacing,
            left,
            right,
            gap_cutoff=gap_cutoff,
            softening=softening,
            dtype=dtype,
            device=device,
        )
        for left in range(particles)
        for right in range(left + 1, particles)
    ]
    return sum_mpos(pair_mpos)


def gap_soft_coulomb_hamiltonian_mpo(
    grid_points: int,
    particles: int,
    spacing: float,
    *,
    gap_cutoff: int | None = None,
    softening: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Return exact kinetic, harmonic, and soft-Coulomb MPOs summed together."""

    return sum_mpos(
        [
            gap_kinetic_harmonic_mpo(
                grid_points,
                particles,
                spacing,
                gap_cutoff=gap_cutoff,
                dtype=dtype,
                device=device,
            ),
            gap_soft_coulomb_mpo(
                grid_points,
                particles,
                spacing,
                gap_cutoff=gap_cutoff,
                softening=softening,
                dtype=dtype,
                device=device,
            ),
        ]
    )


def ordered_values_to_gap_mps(
    values: torch.Tensor,
    grid_points: int,
    particles: int,
    *,
    gap_cutoff: int | None = None,
    max_bond: int | None = None,
    requires_grad: bool = True,
):
    """Convert ordered-sector amplitudes to an exact/truncated fixed-charge gap MPS."""

    MPS, _ = _latticetn_classes()
    tensor = ordered_values_to_gap_tensor(
        values, grid_points, particles, gap_cutoff=gap_cutoff
    )
    cores, ranks, discarded_norm = particle_tensor_to_mps_tensors(
        tensor, max_bond=max_bond
    )
    mps = MPS.from_tensors(
        cores,
        dtype=values.dtype,
        device=values.device,
        requires_grad=requires_grad,
    )
    return mps, ranks, discarded_norm
