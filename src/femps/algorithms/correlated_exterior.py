"""Bounded explicit-correlation prototypes for exterior fermion carriers.

This module deliberately starts with the smallest falsifiable case: two
spinless fermions in a one-dimensional harmonic functional basis.  A symmetric
Gaussian Jastrow factor multiplies an antisymmetric Slater/exterior carrier,
so exchange antisymmetry is exact while the correlation factor is not reduced
to a fixed finite determinant list.

The product Gauss--Hermite grid used here is a deterministic small-system truth
oracle.  It materializes ``Q**2`` coordinate values and is not a production
contraction algorithm.  Its purpose is to establish value, projection,
antisymmetry, and automatic-differentiation checks before any VMC backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class CorrelatedTwoFermionResult:
    """Differentiable observables and structural residuals on a bounded grid."""

    norm: torch.Tensor
    energy: torch.Tensor
    kinetic_energy: torch.Tensor
    trap_energy: torch.Tensor
    interaction_energy: torch.Tensor
    energy_variance: torch.Tensor
    antisymmetry_residual: torch.Tensor
    correlator_symmetry_residual: torch.Tensor
    quadrature_order: int
    materialized_coordinate_values: int


def _validate_carrier(
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    quadrature_order: int,
) -> None:
    if orbitals.ndim != 2 or orbitals.shape[1] != 2 or orbitals.shape[0] < 2:
        raise ValueError("orbitals must have shape (D,2) with D >= 2")
    if orbitals.dtype not in (torch.float64, torch.complex128):
        raise TypeError("orbitals must use float64 or complex128")
    if amplitudes.ndim != 1 or exponents.ndim != 1:
        raise ValueError("correlator amplitudes and exponents must be vectors")
    if amplitudes.shape != exponents.shape:
        raise ValueError("correlator amplitudes and exponents must have equal shape")
    if amplitudes.device != orbitals.device or exponents.device != orbitals.device:
        raise ValueError("carrier and correlator tensors must share a device")
    if amplitudes.dtype != orbitals.dtype or exponents.dtype != orbitals.real.dtype:
        raise TypeError("correlator dtypes must match the carrier real dtype")
    if exponents.numel() and bool(torch.any(exponents <= 0)):
        raise ValueError("Gaussian correlator exponents must be positive")
    if quadrature_order < 4:
        raise ValueError("quadrature_order must be at least four")


def gauss_hermite_rule(
    order: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return physicists' Gauss--Hermite nodes and weights in float64."""

    if order < 1:
        raise ValueError("quadrature order must be positive")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return (
        torch.as_tensor(nodes, dtype=torch.float64, device=device),
        torch.as_tensor(weights, dtype=torch.float64, device=device),
    )


def harmonic_function_values(order: int, nodes: torch.Tensor) -> torch.Tensor:
    """Evaluate normalized unit-frequency harmonic functions ``phi_n(x)``."""

    if order < 1 or nodes.ndim != 1 or nodes.dtype != torch.float64:
        raise ValueError("require positive order and a float64 node vector")
    values = [math.pi ** (-0.25) * torch.exp(-0.5 * nodes.square())]
    if order == 1:
        return torch.stack(values)
    values.append(math.sqrt(2.0) * nodes * values[0])
    for degree in range(1, order - 1):
        values.append(
            math.sqrt(2.0 / (degree + 1)) * nodes * values[degree]
            - math.sqrt(degree / (degree + 1)) * values[degree - 1]
        )
    return torch.stack(values)


def harmonic_function_values_and_derivatives(
    order: int, nodes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``phi_n`` and its first derivative for ``0 <= n < order``."""

    extended = harmonic_function_values(order + 1, nodes)
    derivative = [-math.sqrt(0.5) * extended[1]]
    for degree in range(1, order):
        derivative.append(
            math.sqrt(degree / 2.0) * extended[degree - 1]
            - math.sqrt((degree + 1) / 2.0) * extended[degree + 1]
        )
    return extended[:order], torch.stack(derivative)


def _carrier_grid(
    orbitals: torch.Tensor,
    basis_values: torch.Tensor,
    basis_derivatives: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    values = orbitals.mT @ basis_values.to(dtype=orbitals.dtype)
    normalization = math.sqrt(2.0)
    carrier = (
        values[0, :, None] * values[1, None, :]
        - values[1, :, None] * values[0, None, :]
    ) / normalization
    if basis_derivatives is None:
        return carrier, None, None
    derivatives = orbitals.mT @ basis_derivatives.to(dtype=orbitals.dtype)
    dx1 = (
        derivatives[0, :, None] * values[1, None, :]
        - derivatives[1, :, None] * values[0, None, :]
    ) / normalization
    dx2 = (
        values[0, :, None] * derivatives[1, None, :]
        - values[1, :, None] * derivatives[0, None, :]
    ) / normalization
    return carrier, dx1, dx2


def _correlator_grid(
    nodes: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    separation = nodes[:, None] - nodes[None, :]
    if amplitudes.numel() == 0:
        ones = torch.ones_like(separation, dtype=amplitudes.dtype)
        zeros = torch.zeros_like(ones)
        return ones, zeros, zeros
    separation_typed = separation.to(dtype=amplitudes.dtype)
    gaussian = torch.exp(
        -exponents[:, None, None] * separation_typed.square()[None, :, :]
    )
    log_correlator = torch.einsum("m,mij->ij", amplitudes, gaussian)
    derivative_log = torch.einsum(
        "m,mij->ij",
        amplitudes,
        -2.0
        * exponents[:, None, None]
        * separation_typed[None, :, :]
        * gaussian,
    )
    second_derivative_log = torch.einsum(
        "m,mij->ij",
        amplitudes,
        (
            -2.0 * exponents[:, None, None]
            + 4.0
            * exponents[:, None, None].square()
            * separation_typed[None, :, :].square()
        )
        * gaussian,
    )
    return torch.exp(log_correlator), derivative_log, second_derivative_log


def correlated_two_fermion_observables(
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    *,
    quadrature_order: int,
    omega: float = 1.0,
    coupling: float = 1.0,
    softening: float = 1.0,
) -> CorrelatedTwoFermionResult:
    """Evaluate a Jastrow-correlated exterior carrier by bounded quadrature."""

    _validate_carrier(orbitals, amplitudes, exponents, quadrature_order)
    if not math.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    if not math.isfinite(coupling) or coupling < 0:
        raise ValueError("coupling must be finite and nonnegative")
    if not math.isfinite(softening) or softening <= 0:
        raise ValueError("softening must be finite and positive")

    nodes, weights = gauss_hermite_rule(
        quadrature_order, device=orbitals.device
    )
    basis, basis_derivative = harmonic_function_values_and_derivatives(
        orbitals.shape[0], nodes
    )
    carrier, carrier_dx1, carrier_dx2 = _carrier_grid(
        orbitals, basis, basis_derivative
    )
    assert carrier_dx1 is not None and carrier_dx2 is not None
    levels = torch.arange(
        orbitals.shape[0], dtype=torch.float64, device=orbitals.device
    )
    basis_second_derivative = (
        nodes[None, :].square() - (2.0 * levels[:, None] + 1.0)
    ) * basis
    _, carrier_dx1_dx1, carrier_dx2_dx2 = _carrier_grid(
        orbitals, basis, basis_second_derivative
    )
    assert carrier_dx1_dx1 is not None and carrier_dx2_dx2 is not None
    correlator, derivative_log, second_derivative_log = _correlator_grid(
        nodes, amplitudes, exponents
    )
    wavefunction = correlator * carrier
    derivative_x1 = correlator * (carrier_dx1 + derivative_log * carrier)
    derivative_x2 = correlator * (carrier_dx2 - derivative_log * carrier)

    effective_weight = weights * torch.exp(nodes.square())
    product_weight = effective_weight[:, None] * effective_weight[None, :]
    density = wavefunction.abs().square()
    norm = torch.sum(product_weight * density)
    kinetic = 0.5 * torch.sum(
        product_weight
        * (derivative_x1.abs().square() + derivative_x2.abs().square())
    )
    separation = nodes[:, None] - nodes[None, :]
    trap_potential = 0.5 * omega * omega * (
        nodes[:, None].square() + nodes[None, :].square()
    )
    interaction_potential = coupling * torch.rsqrt(
        separation.square() + softening * softening
    )
    trap = torch.sum(product_weight * trap_potential * density)
    interaction = torch.sum(product_weight * interaction_potential * density)
    energy = (kinetic + trap + interaction) / norm
    correlator_curvature = derivative_log.square() + second_derivative_log
    second_x1 = correlator * (
        carrier_dx1_dx1
        + 2.0 * derivative_log * carrier_dx1
        + correlator_curvature * carrier
    )
    second_x2 = correlator * (
        carrier_dx2_dx2
        - 2.0 * derivative_log * carrier_dx2
        + correlator_curvature * carrier
    )
    h_wavefunction = (
        -0.5 * (second_x1 + second_x2)
        + (trap_potential + interaction_potential) * wavefunction
    )
    variance = torch.sum(
        product_weight * torch.abs(h_wavefunction - energy * wavefunction).square()
    ) / norm

    scale = torch.max(wavefunction.abs()).clamp_min(torch.finfo(torch.float64).tiny)
    antisymmetry = torch.max(torch.abs(wavefunction + wavefunction.mT)) / scale
    correlator_scale = torch.max(correlator.abs()).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    correlator_symmetry = torch.max(torch.abs(correlator - correlator.mT)) / correlator_scale
    return CorrelatedTwoFermionResult(
        norm=norm,
        energy=energy,
        kinetic_energy=kinetic / norm,
        trap_energy=trap / norm,
        interaction_energy=interaction / norm,
        energy_variance=variance.real,
        antisymmetry_residual=antisymmetry,
        correlator_symmetry_residual=correlator_symmetry,
        quadrature_order=quadrature_order,
        materialized_coordinate_values=quadrature_order**2,
    )


def project_correlated_two_fermion_coefficients(
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    *,
    projection_order: int,
    quadrature_order: int,
) -> torch.Tensor:
    """Project the correlated state into ``Lambda^2 V_projection_order``.

    This routine is an explicitly bounded materialization oracle.  It is used
    to measure the Slater rank required by the correlated carrier as the
    functional basis grows; it is not a production contraction path.
    """

    _validate_carrier(orbitals, amplitudes, exponents, quadrature_order)
    if projection_order < orbitals.shape[0]:
        raise ValueError("projection_order must contain the carrier basis")
    nodes, weights = gauss_hermite_rule(
        quadrature_order, device=orbitals.device
    )
    projection_basis = harmonic_function_values(projection_order, nodes).to(
        dtype=orbitals.dtype
    )
    carrier_basis = projection_basis[: orbitals.shape[0]].real
    carrier, _, _ = _carrier_grid(orbitals, carrier_basis)
    correlator, _, _ = _correlator_grid(nodes, amplitudes, exponents)
    wavefunction = correlator * carrier
    effective_weight = weights * torch.exp(nodes.square())
    weighted_basis = projection_basis * effective_weight[None, :]
    return torch.einsum(
        "pi,ij,qj->pq", weighted_basis, wavefunction, weighted_basis
    )
