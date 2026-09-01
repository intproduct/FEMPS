from dataclasses import asdict

import pytest
import torch

from femps.basis import harmonic_hamiltonian
from femps.exterior import agp_exterior_coefficients
from femps.hamiltonians import (
    agp_energy,
    antisymmetric_two_particle_hamiltonian,
    factorize_dense_two_body_operator,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
    soft_coulomb_two_fermion_relative_grid_energy,
)


def test_physical_two_body_svd_reconstructs_exchange_symmetric_tensor() -> None:
    generator = torch.Generator().manual_seed(19)
    raw = torch.randn((4, 4, 4, 4), generator=generator, dtype=torch.float64)
    dense = 0.5 * (raw + raw.permute(1, 0, 3, 2))
    operator, diagnostics = factorize_dense_two_body_operator(dense)
    assert torch.allclose(operator.dense(), dense, atol=2e-13, rtol=2e-13)
    assert diagnostics.retained_rank <= 16
    assert diagnostics.dense_relative_error < 2e-14
    assert diagnostics.particle_exchange_residual < 2e-14


@pytest.mark.parametrize("dimension", [10, 12])
def test_physical_soft_coulomb_factorization_stays_accurate_at_larger_basis(
    dimension: int,
) -> None:
    operator, diagnostics = soft_coulomb_operator(
        dimension,
        quadrature_order=128,
        relative_threshold=1e-13,
        factorization_backend="physical",
    )
    direct = soft_coulomb_dense_quadrature(dimension, quadrature_order=128)
    relative_error = torch.linalg.vector_norm(operator.dense() - direct) / torch.linalg.vector_norm(direct)
    assert diagnostics.factorization_backend == "physical_operator_svd"
    assert diagnostics.dense_relative_factorization_error < 1e-11
    assert relative_error < 1e-11


def test_soft_coulomb_factorization_matches_direct_quadrature() -> None:
    operator, diagnostics = soft_coulomb_operator(
        5,
        quadrature_order=32,
        coupling=0.7,
        softening=0.8,
        relative_threshold=0.0,
    )
    direct = soft_coulomb_dense_quadrature(
        5, quadrature_order=32, coupling=0.7, softening=0.8
    )
    assert torch.allclose(operator.dense(), direct, atol=2e-13, rtol=2e-13)
    assert diagnostics.retained_rank == 32
    assert diagnostics.dense_relative_factorization_error < 2e-14
    assert diagnostics.dense_hermiticity_residual < 2e-14
    assert diagnostics.exchange_symmetry_residual < 2e-14
    assert asdict(diagnostics)["softening"] == 0.8


def test_soft_coulomb_quadrature_converges() -> None:
    coarse = soft_coulomb_dense_quadrature(
        4, quadrature_order=24, softening=1.0
    )
    medium = soft_coulomb_dense_quadrature(
        4, quadrature_order=40, softening=1.0
    )
    fine = soft_coulomb_dense_quadrature(
        4, quadrature_order=64, softening=1.0
    )
    coarse_error = torch.linalg.vector_norm(coarse - fine)
    medium_error = torch.linalg.vector_norm(medium - fine)
    assert medium_error < coarse_error
    assert medium_error / torch.linalg.vector_norm(fine) < 2e-5


def test_relative_grid_energy_converges_at_second_order() -> None:
    coarse = soft_coulomb_two_fermion_relative_grid_energy(intervals=80)
    medium = soft_coulomb_two_fermion_relative_grid_energy(intervals=120)
    fine = soft_coulomb_two_fermion_relative_grid_energy(intervals=180)
    assert abs(fine - medium) < abs(medium - coarse)
    assert 2.5 < fine < 2.6


def test_soft_coulomb_agp_energy_matches_exterior_truth_and_gradient() -> None:
    dimension = 5
    generator = torch.Generator().manual_seed(71)
    raw_real = torch.randn(
        dimension, dimension, generator=generator, dtype=torch.float64
    )
    raw = torch.complex(raw_real, torch.zeros_like(raw_real)).requires_grad_()
    pair = raw - raw.transpose(0, 1)
    one_body = harmonic_hamiltonian(dimension, dtype=torch.complex128)
    operator, _ = soft_coulomb_operator(
        dimension,
        quadrature_order=48,
        coupling=0.4,
        softening=1.0,
        relative_threshold=0.0,
    )
    polynomial = agp_energy(pair, 1, one_body, operator)
    exterior_hamiltonian = antisymmetric_two_particle_hamiltonian(one_body, operator)
    state = agp_exterior_coefficients(pair, 1)
    explicit = (
        torch.vdot(state, exterior_hamiltonian @ state) / torch.vdot(state, state)
    ).real
    polynomial_gradient = torch.autograd.grad(polynomial, raw, retain_graph=True)[0]
    explicit_gradient = torch.autograd.grad(explicit, raw)[0]
    assert torch.allclose(polynomial, explicit, atol=2e-11, rtol=2e-11)
    assert torch.allclose(
        polynomial_gradient, explicit_gradient, atol=3e-10, rtol=3e-10
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quadrature_order": 0},
        {"quadrature_order": 8, "softening": 0.0},
        {"quadrature_order": 8, "coupling": -1.0},
        {"quadrature_order": 8, "relative_threshold": 1.0},
    ],
)
def test_soft_coulomb_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        soft_coulomb_operator(3, **kwargs)
