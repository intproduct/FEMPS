"""Native functional-basis MPOs in center-of-mass/positive-distance variables."""

from __future__ import annotations

import math

import torch

from femps.basis.dirichlet_sine import (
    dirichlet_sine_derivative_matrix,
    dirichlet_sine_negative_second_derivative_matrix,
    dirichlet_sine_position_matrix,
    dirichlet_sine_position_squared_matrix,
)
from femps.basis.harmonic import (
    derivative_matrix,
    negative_second_derivative_matrix,
    position_matrix,
    position_squared_matrix,
)
from femps.baselines.ordered_distance_mpo import product_sum_mpo, sum_mpos
from femps.ordered_continuous import (
    ordered_harmonic_metric,
    ordered_kinetic_metric,
)


def ordered_continuous_local_operators(
    particles: int,
    basis_order: int,
    distance_length: float,
    *,
    distance_basis: str = "dirichlet_sine",
    distance_scale_ratio: float = 2.0,
    center_of_mass_length: float | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Return scaled local operators for the COM site and distance sites."""

    if particles < 1 or basis_order < 1 or distance_length <= 0:
        raise ValueError("require positive particles, basis order, and distance length")
    com_length = (
        1 / math.sqrt(particles)
        if center_of_mass_length is None
        else center_of_mass_length
    )
    if com_length <= 0:
        raise ValueError("center_of_mass_length must be positive")
    identity = torch.eye(basis_order, dtype=dtype, device=device)
    com = {
        "identity": identity,
        "derivative": derivative_matrix(
            basis_order, dtype=dtype, device=device
        )
        / com_length,
        "negative_second_derivative": negative_second_derivative_matrix(
            basis_order, dtype=dtype, device=device
        )
        / com_length**2,
        "position": com_length
        * position_matrix(basis_order, dtype=dtype, device=device),
        "position_squared": com_length**2
        * position_squared_matrix(basis_order, dtype=dtype, device=device),
    }
    if distance_basis == "dirichlet_sine":
        distance = {
            "identity": identity,
            "derivative": dirichlet_sine_derivative_matrix(
                basis_order, distance_length, dtype=dtype, device=device
            ),
            "negative_second_derivative": (
                dirichlet_sine_negative_second_derivative_matrix(
                    basis_order, distance_length, dtype=dtype, device=device
                )
            ),
            "position": dirichlet_sine_position_matrix(
                basis_order, distance_length, dtype=dtype, device=device
            ),
            "position_squared": dirichlet_sine_position_squared_matrix(
                basis_order, distance_length, dtype=dtype, device=device
            ),
        }
    elif distance_basis == "odd_hermite":
        from femps.basis.odd_hermite import (
            odd_hermite_derivative_matrix,
            odd_hermite_negative_second_derivative_matrix,
            odd_hermite_position_matrix,
            odd_hermite_position_squared_matrix,
        )

        distance = {
            "identity": identity,
            "derivative": odd_hermite_derivative_matrix(
                basis_order, distance_length, dtype=dtype, device=device
            ),
            "negative_second_derivative": (
                odd_hermite_negative_second_derivative_matrix(
                    basis_order, distance_length, dtype=dtype, device=device
                )
            ),
            "position": odd_hermite_position_matrix(
                basis_order, distance_length, dtype=dtype, device=device
            ),
            "position_squared": odd_hermite_position_squared_matrix(
                basis_order, distance_length, dtype=dtype, device=device
            ),
        }
    elif distance_basis == "multiscale_odd_hermite":
        from femps.basis.multiscale_odd_hermite import (
            multiscale_odd_hermite_derivative_matrix,
            multiscale_odd_hermite_negative_second_derivative_matrix,
            multiscale_odd_hermite_position_matrix,
            multiscale_odd_hermite_position_squared_matrix,
        )

        distance = {
            "identity": identity,
            "derivative": multiscale_odd_hermite_derivative_matrix(
                basis_order,
                distance_length,
                distance_scale_ratio,
                dtype=dtype,
                device=device,
            ),
            "negative_second_derivative": (
                multiscale_odd_hermite_negative_second_derivative_matrix(
                    basis_order,
                    distance_length,
                    distance_scale_ratio,
                    dtype=dtype,
                    device=device,
                )
            ),
            "position": multiscale_odd_hermite_position_matrix(
                basis_order,
                distance_length,
                distance_scale_ratio,
                dtype=dtype,
                device=device,
            ),
            "position_squared": (
                multiscale_odd_hermite_position_squared_matrix(
                    basis_order,
                    distance_length,
                    distance_scale_ratio,
                    dtype=dtype,
                    device=device,
                )
            ),
        }
    else:
        raise ValueError(
            "distance_basis must be 'dirichlet_sine', 'odd_hermite', or "
            "'multiscale_odd_hermite'"
        )
    return [com] + [distance for _ in range(particles - 1)]


def ordered_continuous_noninteracting_terms(
    particles: int,
    basis_order: int,
    distance_length: float,
    *,
    distance_basis: str = "dirichlet_sine",
    distance_scale_ratio: float = 2.0,
    center_of_mass_length: float | None = None,
    omega: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> list[tuple[torch.Tensor, ...]]:
    """Return exact Galerkin product terms for kinetic plus harmonic trap."""

    if omega <= 0:
        raise ValueError("omega must be positive")
    operators = ordered_continuous_local_operators(
        particles,
        basis_order,
        distance_length,
        distance_basis=distance_basis,
        distance_scale_ratio=distance_scale_ratio,
        center_of_mass_length=center_of_mass_length,
        dtype=dtype,
        device=device,
    )
    kinetic_metric = ordered_kinetic_metric(
        particles, dtype=dtype, device=device
    )
    harmonic_metric = ordered_harmonic_metric(
        particles, dtype=dtype, device=device
    )
    identities = [local["identity"] for local in operators]
    terms: list[tuple[torch.Tensor, ...]] = []

    def add_term(assignments: dict[int, torch.Tensor], coefficient) -> None:
        local = list(identities)
        for site, operator in assignments.items():
            local[site] = operator
        local[0] = coefficient * local[0]
        terms.append(tuple(local))

    for site in range(particles):
        add_term(
            {site: operators[site]["negative_second_derivative"]},
            0.5 * kinetic_metric[site, site],
        )
        add_term(
            {site: operators[site]["position_squared"]},
            0.5 * omega**2 * harmonic_metric[site, site],
        )
    for left in range(particles):
        for right in range(left + 1, particles):
            if kinetic_metric[left, right] != 0:
                add_term(
                    {
                        left: operators[left]["derivative"],
                        right: operators[right]["derivative"],
                    },
                    -kinetic_metric[left, right],
                )
            if harmonic_metric[left, right] != 0:
                add_term(
                    {
                        left: operators[left]["position"],
                        right: operators[right]["position"],
                    },
                    omega**2 * harmonic_metric[left, right],
                )
    return terms


def ordered_continuous_noninteracting_mpo(
    particles: int,
    basis_order: int,
    distance_length: float,
    *,
    distance_basis: str = "dirichlet_sine",
    distance_scale_ratio: float = 2.0,
    center_of_mass_length: float | None = None,
    omega: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Return the native latticeTN MPO for the continuous Galerkin operator."""

    return product_sum_mpo(
        ordered_continuous_noninteracting_terms(
            particles,
            basis_order,
            distance_length,
            distance_basis=distance_basis,
            distance_scale_ratio=distance_scale_ratio,
            center_of_mass_length=center_of_mass_length,
            omega=omega,
            dtype=dtype,
            device=device,
        )
    )


def ordered_continuous_soft_coulomb_hamiltonian_mpo(
    particles: int,
    basis_order: int,
    distance_length: float,
    interaction_degree: int,
    *,
    distance_basis: str = "dirichlet_sine",
    distance_scale_ratio: float = 2.0,
    center_of_mass_length: float | None = None,
    omega: float = 1.0,
    coupling: float = 1.0,
    softening: float = 1.0,
    interaction_quadrature_order: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    """Return kinetic, harmonic, and controlled soft-Coulomb functional MPO."""

    if coupling < 0:
        raise ValueError("coupling must be nonnegative")
    noninteracting = ordered_continuous_noninteracting_mpo(
        particles,
        basis_order,
        distance_length,
        distance_basis=distance_basis,
        distance_scale_ratio=distance_scale_ratio,
        center_of_mass_length=center_of_mass_length,
        omega=omega,
        dtype=dtype,
        device=device,
    )
    if coupling == 0:
        return noninteracting
    if distance_basis != "dirichlet_sine":
        raise ValueError(
            "soft-Coulomb interval polynomial currently requires dirichlet_sine"
        )
    from femps.baselines.ordered_continuous_interaction import (
        ordered_continuous_soft_coulomb_mpo,
    )

    interaction = ordered_continuous_soft_coulomb_mpo(
        particles,
        basis_order,
        distance_length,
        interaction_degree,
        softening=softening,
        quadrature_order=interaction_quadrature_order,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        interaction.tensors[0].mul_(coupling)
    return sum_mpos([noninteracting, interaction])
