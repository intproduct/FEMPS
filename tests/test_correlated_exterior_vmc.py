from pathlib import Path

import torch

from femps.algorithms import (
    CorrelatedExteriorVMCConfig,
    canonical_exterior_carrier,
    canonical_lowest_orbitals,
    correlated_exterior_local_energy,
    correlated_exterior_wavefunction_value,
    gaussian_pair_log_correlator,
    run_correlated_exterior_vmc,
    sampled_antisymmetry_residual,
)
from femps.algorithms.correlated_exterior import (
    correlated_two_fermion_observables,
    gauss_hermite_rule,
)


def _empty() -> torch.Tensor:
    return torch.empty(0, dtype=torch.float64)


def test_gaussian_pair_coordinate_derivatives_match_autograd() -> None:
    positions = torch.tensor([-1.1, 0.2, 1.4], dtype=torch.float64, requires_grad=True)
    amplitudes = torch.tensor([-0.2, 0.05], dtype=torch.float64)
    exponents = torch.tensor([0.5, 2.0], dtype=torch.float64)
    value, gradient, laplacian = gaussian_pair_log_correlator(
        positions, amplitudes, exponents
    )
    autograd_gradient = torch.autograd.grad(value, positions, create_graph=True)[0]
    autograd_laplacian = torch.stack(
        [
            torch.autograd.grad(
                autograd_gradient[index], positions, retain_graph=True
            )[0][index]
            for index in range(positions.numel())
        ]
    )
    torch.testing.assert_close(gradient, autograd_gradient, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(
        laplacian, autograd_laplacian, atol=3e-14, rtol=3e-14
    )


def test_n4_noninteracting_slater_has_exact_local_energy_and_swap_sign() -> None:
    orbitals = canonical_lowest_orbitals(4, 4)
    positions = torch.tensor(
        [
            [-1.2, -0.4, 0.5, 1.4],
            [0.1, -1.0, 1.1, 2.0],
            [-2.0, -0.7, 0.3, 1.8],
        ],
        dtype=torch.float64,
    )
    result = correlated_exterior_local_energy(
        orbitals, _empty(), _empty(), positions, coupling=0.0
    )
    torch.testing.assert_close(
        result.local_energy,
        torch.full((3,), 8.0, dtype=torch.float64),
        atol=2e-13,
        rtol=2e-13,
    )
    torch.testing.assert_close(
        result.kinetic_energy + result.trap_energy,
        result.local_energy,
        atol=2e-14,
        rtol=2e-14,
    )
    symmetry = sampled_antisymmetry_residual(
        orbitals, _empty(), _empty(), positions
    )
    assert symmetry["antisymmetry_residual"] <= 2e-15
    assert symmetry["correlator_symmetry_residual"] == 0.0

    swapped = positions.clone()
    swapped[:, 0], swapped[:, 1] = positions[:, 1], positions[:, 0]
    value = correlated_exterior_wavefunction_value(
        orbitals, _empty(), _empty(), positions
    )
    swapped_value = correlated_exterior_wavefunction_value(
        orbitals, _empty(), _empty(), swapped
    )
    torch.testing.assert_close(swapped_value, -value, atol=2e-15, rtol=2e-15)


def test_n2_local_energy_integrates_to_existing_quadrature_truth() -> None:
    orbitals = canonical_lowest_orbitals(4, 2)
    amplitudes = torch.tensor([-0.2, 0.05], dtype=torch.float64)
    exponents = torch.tensor([0.25, 1.0], dtype=torch.float64)
    order = 48
    truth = correlated_two_fermion_observables(
        orbitals,
        amplitudes,
        exponents,
        quadrature_order=order,
        coupling=1.0,
        softening=1.0,
    )
    nodes, weights = gauss_hermite_rule(order)
    x1, x2 = torch.meshgrid(nodes, nodes, indexing="ij")
    positions = torch.stack((x1.reshape(-1), x2.reshape(-1)), dim=-1)
    wavefunction = correlated_exterior_wavefunction_value(
        orbitals, amplitudes, exponents, positions
    )
    nonnode = torch.abs(positions[:, 0] - positions[:, 1]) > 1e-14
    local = correlated_exterior_local_energy(
        orbitals,
        amplitudes,
        exponents,
        positions[nonnode],
        coupling=1.0,
        softening=1.0,
    )
    effective_weight = weights * torch.exp(nodes.square())
    product_weight = (effective_weight[:, None] * effective_weight[None, :]).reshape(-1)
    density_weight = product_weight * wavefunction.square()
    energy = torch.sum(
        density_weight[nonnode] * local.local_energy
    ) / torch.sum(density_weight)
    torch.testing.assert_close(energy, truth.energy, atol=2e-12, rtol=2e-12)


def test_vmc_checkpoint_resume_matches_clean_run(tmp_path: Path) -> None:
    config = CorrelatedExteriorVMCConfig(
        particles=2,
        chains=4,
        burn_in_sweeps=5,
        samples_per_chain=12,
        thinning_sweeps=2,
        proposal_scale=0.6,
        seed=43001,
        max_autocorrelation_lag=5,
        checkpoint_every=3,
        coupling=0.0,
    )
    orbitals = canonical_lowest_orbitals(2, 2)
    checkpoint = tmp_path / "vmc.pt"
    partial = run_correlated_exterior_vmc(
        config,
        orbitals,
        _empty(),
        _empty(),
        checkpoint_path=checkpoint,
        max_samples_this_call=5,
    )
    assert partial["completed"] is False
    assert partial["completed_samples_per_chain"] == 5
    resumed = run_correlated_exterior_vmc(
        config,
        orbitals,
        _empty(),
        _empty(),
        checkpoint_path=checkpoint,
        resume=True,
    )
    clean = run_correlated_exterior_vmc(config, orbitals, _empty(), _empty())
    assert resumed["completed"] is True
    assert resumed["accepted_proposals"] == clean["accepted_proposals"]
    assert resumed["total_proposals"] == clean["total_proposals"]
    assert resumed["acceptance_rate"] == clean["acceptance_rate"]
    assert torch.equal(resumed["samples"], clean["samples"])
    assert resumed["energy"] == clean["energy"] == 2.0
    assert resumed["energy_variance"] < 1e-27
    assert resumed["symmetry"]["antisymmetry_residual"] <= 2e-15


def test_general_carrier_qr_is_orthonormal() -> None:
    raw = torch.randn(
        (7, 4), generator=torch.Generator().manual_seed(43002), dtype=torch.float64
    )
    orbitals = canonical_exterior_carrier(raw)
    torch.testing.assert_close(
        orbitals.mT @ orbitals,
        torch.eye(4, dtype=torch.float64),
        atol=3e-14,
        rtol=3e-14,
    )
