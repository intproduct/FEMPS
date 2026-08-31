"""Exact finite-box gap coordinates for the ordered fermion sector.

For ``N`` strictly ordered particles on ``L`` grid points, ``N+1`` nonnegative
gaps sum to ``L-N``.  The two boundary gaps count empty sites outside the
outer particles and the interior gaps count empty sites between neighbors.
"""

from __future__ import annotations

import itertools
import math

import torch

from .ordered_sector import ordered_configurations


def gap_configurations(
    grid_points: int,
    particles: int,
    *,
    gap_cutoff: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return all ``N+1`` gaps with total empty-site charge ``L-N``."""

    if particles < 1 or grid_points < particles:
        raise ValueError("require 1 <= particles <= grid_points")
    holes = grid_points - particles

    def compositions(total: int, parts: int):
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for tail in compositions(total - first, parts - 1):
                yield (first,) + tail

    cutoff = holes if gap_cutoff is None else gap_cutoff
    if not 0 <= cutoff <= holes:
        raise ValueError("gap_cutoff must satisfy 0 <= gap_cutoff <= L-N")
    result = tuple(
        gaps
        for gaps in compositions(holes, particles + 1)
        if max(gaps) <= cutoff
    )
    if gap_cutoff is None and len(result) != math.comb(grid_points, particles):
        raise AssertionError("gap/ordered dimension identity failed")
    return result


def ordered_configuration_to_gaps(
    configuration: tuple[int, ...], grid_points: int
) -> tuple[int, ...]:
    """Map ``0 <= x_1 < ... < x_N < L`` to boundary/interior gaps."""

    particles = len(configuration)
    if particles < 1 or grid_points < particles:
        raise ValueError("require a nonempty configuration with N <= L")
    if any(
        configuration[index] >= configuration[index + 1]
        for index in range(particles - 1)
    ) or configuration[0] < 0 or configuration[-1] >= grid_points:
        raise ValueError("configuration must be strictly ordered inside the grid")
    gaps = [configuration[0]]
    gaps.extend(
        configuration[index] - configuration[index - 1] - 1
        for index in range(1, particles)
    )
    gaps.append(grid_points - 1 - configuration[-1])
    return tuple(gaps)


def gap_configuration_to_ordered(
    gaps: tuple[int, ...], grid_points: int
) -> tuple[int, ...]:
    """Invert :func:`ordered_configuration_to_gaps`."""

    if len(gaps) < 2 or any(gap < 0 for gap in gaps):
        raise ValueError("gaps must contain N+1 nonnegative integers")
    particles = len(gaps) - 1
    if sum(gaps) != grid_points - particles:
        raise ValueError("gaps must sum to L-N")
    positions = [gaps[0]]
    for interior_gap in gaps[1:-1]:
        positions.append(positions[-1] + 1 + interior_gap)
    if positions[-1] + gaps[-1] != grid_points - 1:
        raise AssertionError("right boundary gap is inconsistent")
    return tuple(positions)


def ordered_values_to_gap_values(
    values: torch.Tensor,
    grid_points: int,
    particles: int,
    *,
    gap_cutoff: int | None = None,
) -> torch.Tensor:
    """Permute lexicographic ordered-sector values into gap-composition order."""

    ordered = ordered_configurations(grid_points, particles)
    if values.ndim != 1 or values.numel() != len(ordered):
        raise ValueError("values must contain one amplitude per ordered configuration")
    lookup = {configuration: index for index, configuration in enumerate(ordered)}
    return torch.stack(
        [
            values[lookup[gap_configuration_to_ordered(gaps, grid_points)]]
            for gaps in gap_configurations(
                grid_points, particles, gap_cutoff=gap_cutoff
            )
        ]
    )


def gap_values_to_ordered_values(
    values: torch.Tensor,
    grid_points: int,
    particles: int,
    *,
    gap_cutoff: int | None = None,
) -> torch.Tensor:
    """Invert :func:`ordered_values_to_gap_values`."""

    gaps = gap_configurations(grid_points, particles, gap_cutoff=gap_cutoff)
    if values.ndim != 1 or values.numel() != len(gaps):
        raise ValueError("values must contain one amplitude per gap configuration")
    ordered = ordered_configurations(grid_points, particles)
    ordered_lookup = {configuration: index for index, configuration in enumerate(ordered)}
    result = torch.zeros(
        len(ordered), dtype=values.dtype, device=values.device
    )
    for index, gap_configuration in enumerate(gaps):
        ordered_configuration = gap_configuration_to_ordered(
            gap_configuration, grid_points
        )
        result[ordered_lookup[ordered_configuration]] = values[index]
    return result


def ordered_values_to_gap_tensor(
    values: torch.Tensor,
    grid_points: int,
    particles: int,
    *,
    gap_cutoff: int | None = None,
) -> torch.Tensor:
    """Scatter ordered amplitudes into a fixed-charge ``(L-N+1)^(N+1)`` tensor."""

    gap_values = ordered_values_to_gap_values(
        values, grid_points, particles, gap_cutoff=gap_cutoff
    )
    configurations = gap_configurations(
        grid_points, particles, gap_cutoff=gap_cutoff
    )
    holes = grid_points - particles
    cutoff = holes if gap_cutoff is None else gap_cutoff
    local_dimension = cutoff + 1
    tensor = torch.zeros(
        (local_dimension,) * (particles + 1),
        dtype=values.dtype,
        device=values.device,
    )
    tensor[tuple(zip(*configurations, strict=True))] = gap_values
    return tensor


def gap_tensor_to_ordered_values(
    tensor: torch.Tensor,
    grid_points: int,
    particles: int,
    *,
    gap_cutoff: int | None = None,
) -> torch.Tensor:
    """Gather the fixed-charge gap sector and return lexicographic ordered values."""

    holes = grid_points - particles
    cutoff = holes if gap_cutoff is None else gap_cutoff
    local_dimension = cutoff + 1
    if tensor.shape != (local_dimension,) * (particles + 1):
        raise ValueError("gap tensor has the wrong shape")
    configurations = gap_configurations(
        grid_points, particles, gap_cutoff=gap_cutoff
    )
    gap_values = tensor[tuple(zip(*configurations, strict=True))]
    return gap_values_to_ordered_values(
        gap_values, grid_points, particles, gap_cutoff=gap_cutoff
    )


def ordered_hamiltonian_to_gap_basis(
    hamiltonian: torch.Tensor,
    grid_points: int,
    particles: int,
    *,
    gap_cutoff: int | None = None,
) -> torch.Tensor:
    """Permute an ordered-sector matrix into the gap-composition basis."""

    ordered = ordered_configurations(grid_points, particles)
    if hamiltonian.shape != (len(ordered), len(ordered)):
        raise ValueError("hamiltonian has the wrong ordered-sector shape")
    lookup = {configuration: index for index, configuration in enumerate(ordered)}
    permutation = torch.tensor(
        [
            lookup[gap_configuration_to_ordered(gaps, grid_points)]
            for gaps in gap_configurations(
                grid_points, particles, gap_cutoff=gap_cutoff
            )
        ],
        dtype=torch.long,
        device=hamiltonian.device,
    )
    return hamiltonian[permutation][:, permutation]


def gap_kinetic_hamiltonian(
    grid_points: int,
    particles: int,
    spacing: float,
    *,
    gap_cutoff: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the finite-difference kinetic operator in the fixed-charge gap basis."""

    if spacing <= 0:
        raise ValueError("spacing must be positive")
    configurations = gap_configurations(
        grid_points, particles, gap_cutoff=gap_cutoff
    )
    lookup = {configuration: index for index, configuration in enumerate(configurations)}
    hamiltonian = torch.zeros(
        len(configurations), len(configurations), dtype=dtype, device=device
    )
    hopping = -0.5 / spacing**2
    diagonal = particles / spacing**2
    for column, gaps in enumerate(configurations):
        hamiltonian[column, column] = diagonal
        for particle in range(particles):
            if gaps[particle + 1] > 0:
                moved = list(gaps)
                moved[particle] += 1
                moved[particle + 1] -= 1
                target = tuple(moved)
                if target in lookup:
                    hamiltonian[lookup[target], column] += hopping
            if gaps[particle] > 0:
                moved = list(gaps)
                moved[particle] -= 1
                moved[particle + 1] += 1
                target = tuple(moved)
                if target in lookup:
                    hamiltonian[lookup[target], column] += hopping
    return hamiltonian


def gap_coordinate_positions(
    gaps: tuple[int, ...], grid_points: int, spacing: float
) -> tuple[float, ...]:
    """Return centered physical coordinates represented by a gap tuple."""

    ordered = gap_configuration_to_ordered(gaps, grid_points)
    center = (grid_points - 1) / 2
    return tuple(spacing * (position - center) for position in ordered)


def gap_diagonal_potential(
    grid_points: int,
    particles: int,
    spacing: float,
    *,
    gap_cutoff: int | None = None,
    harmonic: bool = True,
    soft_coulomb: bool = False,
    softening: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return harmonic and/or soft-Coulomb diagonal energies in gap order."""

    if spacing <= 0 or softening <= 0:
        raise ValueError("spacing and softening must be positive")
    entries = []
    for gaps in gap_configurations(
        grid_points, particles, gap_cutoff=gap_cutoff
    ):
        positions = gap_coordinate_positions(gaps, grid_points, spacing)
        value = 0.0
        if harmonic:
            value += 0.5 * sum(position**2 for position in positions)
        if soft_coulomb:
            value += sum(
                1 / math.sqrt((positions[right] - positions[left]) ** 2 + softening**2)
                for left, right in itertools.combinations(range(particles), 2)
            )
        entries.append(value)
    return torch.tensor(entries, dtype=dtype, device=device)


def gap_hamiltonian(
    grid_points: int,
    particles: int,
    spacing: float,
    *,
    gap_cutoff: int | None = None,
    soft_coulomb: bool = False,
    softening: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return kinetic plus harmonic and optional soft-Coulomb gap Hamiltonian."""

    kinetic = gap_kinetic_hamiltonian(
        grid_points,
        particles,
        spacing,
        gap_cutoff=gap_cutoff,
        dtype=dtype,
        device=device,
    )
    diagonal = gap_diagonal_potential(
        grid_points,
        particles,
        spacing,
        gap_cutoff=gap_cutoff,
        soft_coulomb=soft_coulomb,
        softening=softening,
        dtype=dtype,
        device=device,
    )
    return kinetic + torch.diag(diagonal)
