import torch

from femps.algorithms.correlated_exterior import (
    CorrelatedExteriorConfig,
    canonical_two_orbital_carrier,
    correlated_two_fermion_observables,
    harmonic_function_values_and_derivatives,
    project_correlated_two_fermion_coefficients,
    run_correlated_exterior_optimization,
)


def _canonical_two_fermion_carrier(order: int = 4) -> torch.Tensor:
    orbitals = torch.zeros((order, 2), dtype=torch.float64)
    orbitals[0, 0] = 1.0
    orbitals[1, 1] = 1.0
    return orbitals


def test_harmonic_function_derivative_matches_autograd() -> None:
    nodes = torch.linspace(-2.0, 2.0, 17, dtype=torch.float64)
    nodes_ad = nodes.clone().requires_grad_(True)
    values, derivatives = harmonic_function_values_and_derivatives(6, nodes_ad)
    for degree in range(6):
        gradient = torch.autograd.grad(values[degree].sum(), nodes_ad, retain_graph=True)[0]
        torch.testing.assert_close(gradient, derivatives[degree], atol=2e-14, rtol=2e-14)


def test_uncorrelated_carrier_recovers_two_fermion_harmonic_ground_state() -> None:
    result = correlated_two_fermion_observables(
        _canonical_two_fermion_carrier(),
        torch.empty(0, dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
        quadrature_order=48,
        coupling=0.0,
    )
    torch.testing.assert_close(
        result.norm, torch.tensor(1.0, dtype=torch.float64), atol=2e-14, rtol=2e-14
    )
    torch.testing.assert_close(
        result.energy, torch.tensor(2.0, dtype=torch.float64), atol=2e-14, rtol=2e-14
    )
    assert result.energy_variance < 1e-26
    assert result.materialized_coordinate_values == 48**2
    assert result.antisymmetry_residual == 0
    assert result.correlator_symmetry_residual == 0


def test_symmetric_correlator_preserves_antisymmetry_and_ad_gradient() -> None:
    orbitals = _canonical_two_fermion_carrier()
    amplitudes = torch.tensor([0.2, -0.1], dtype=torch.float64, requires_grad=True)
    exponents = torch.tensor([0.5, 2.0], dtype=torch.float64)
    result = correlated_two_fermion_observables(
        orbitals,
        amplitudes,
        exponents,
        quadrature_order=64,
        coupling=1.0,
        softening=1.0,
    )
    gradient = torch.autograd.grad(result.energy, amplitudes)[0]
    assert torch.all(torch.isfinite(gradient))
    assert result.energy_variance > 0
    assert result.antisymmetry_residual == 0
    assert result.correlator_symmetry_residual == 0

    step = 1e-5
    plus = amplitudes.detach().clone()
    minus = amplitudes.detach().clone()
    plus[0] += step
    minus[0] -= step
    energy_plus = correlated_two_fermion_observables(
        orbitals,
        plus,
        exponents,
        quadrature_order=64,
        coupling=1.0,
    ).energy
    energy_minus = correlated_two_fermion_observables(
        orbitals,
        minus,
        exponents,
        quadrature_order=64,
        coupling=1.0,
    ).energy
    finite_difference = (energy_plus - energy_minus) / (2.0 * step)
    torch.testing.assert_close(gradient[0], finite_difference, atol=2e-9, rtol=2e-8)


def test_nontrivial_correlator_exceeds_single_slater_rank_in_projection() -> None:
    coefficients = project_correlated_two_fermion_coefficients(
        _canonical_two_fermion_carrier(),
        torch.tensor([0.2], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        projection_order=8,
        quadrature_order=80,
    )
    relative_skew_residual = torch.linalg.vector_norm(
        coefficients + coefficients.mT
    ) / torch.linalg.vector_norm(coefficients)
    assert relative_skew_residual < 1e-13
    assert torch.linalg.matrix_rank(coefficients, tol=1e-10) >= 6


def test_uncorrelated_projection_materializes_the_exterior_carrier() -> None:
    coefficients = project_correlated_two_fermion_coefficients(
        _canonical_two_fermion_carrier(order=2),
        torch.empty(0, dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
        projection_order=2,
        quadrature_order=48,
    )
    expected = torch.tensor(
        [[0.0, 2.0**-0.5], [-2.0**-0.5, 0.0]], dtype=torch.float64
    )
    torch.testing.assert_close(coefficients, expected, atol=3e-14, rtol=3e-14)


def test_carrier_qr_gradient_matches_central_difference() -> None:
    raw = _canonical_two_fermion_carrier().clone()
    raw[2, 0] = 0.07
    raw.requires_grad_(True)
    amplitudes = torch.tensor([-0.1], dtype=torch.float64)
    exponents = torch.tensor([1.0], dtype=torch.float64)

    def energy(value: torch.Tensor) -> torch.Tensor:
        return correlated_two_fermion_observables(
            canonical_two_orbital_carrier(value),
            amplitudes,
            exponents,
            quadrature_order=48,
        ).energy

    gradient = torch.autograd.grad(energy(raw), raw)[0]
    step = 1e-5
    plus = raw.detach().clone()
    minus = raw.detach().clone()
    plus[2, 0] += step
    minus[2, 0] -= step
    finite_difference = (energy(plus) - energy(minus)) / (2.0 * step)
    torch.testing.assert_close(
        gradient[2, 0], finite_difference, atol=3e-9, rtol=3e-8
    )


def test_bounded_correlated_optimizer_is_seeded_and_nonworsening(tmp_path) -> None:
    config = CorrelatedExteriorConfig(
        basis_order=3,
        exponents=(1.0,),
        seed=40001,
        quadrature_order=32,
        adam_steps=3,
        lbfgs_steps=2,
        record_points=2,
    )
    checkpoint = tmp_path / "correlated.pt"
    first = run_correlated_exterior_optimization(config, checkpoint_path=checkpoint)
    second = run_correlated_exterior_optimization(config)
    assert checkpoint.exists()
    assert first["energy"] <= first["trace"][0]["energy"] + 1e-12
    assert first["antisymmetry_residual"] <= 1e-13
    assert first["correlator_symmetry_residual"] <= 1e-13
    torch.testing.assert_close(first["raw_carrier"], second["raw_carrier"])
    torch.testing.assert_close(first["amplitudes"], second["amplitudes"])
    assert first["optimized_real_parameter_count"] == 7


def test_correlated_carrier_rejects_nonsymmetric_parameter_shapes() -> None:
    with torch.no_grad():
        try:
            correlated_two_fermion_observables(
                _canonical_two_fermion_carrier(),
                torch.zeros(2, dtype=torch.float64),
                torch.ones(1, dtype=torch.float64),
                quadrature_order=16,
            )
        except ValueError as error:
            assert "equal shape" in str(error)
        else:
            raise AssertionError("mismatched correlator shapes must fail")
