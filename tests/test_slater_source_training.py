from dataclasses import replace
from pathlib import Path

import pytest
import torch

from femps.algorithms import (
    AdaptiveDiagonalPathStageConfig,
    SlaterSourceOptimizerConfig,
    SlaterSourceSolverConfig,
    canonical_lowest_slater,
    load_adaptive_diagonal_path_checkpoint,
    load_slater_source_command_config,
    load_slater_source_checkpoint,
    run_slater_source_adaptive_solver,
    solve_generalized_hermitian,
    validate_slater_source_result,
)
from femps.exterior import (
    diagonal_path_exterior_coefficients,
    diagonal_path_hamiltonian_matrices,
)
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian,
    harmonic_pair_hamiltonian,
    soft_coulomb_operator,
)


def _optimizer(seed: int) -> SlaterSourceOptimizerConfig:
    return SlaterSourceOptimizerConfig(
        seed=seed,
        steps=3,
        learning_rate=2e-3,
        final_learning_rate=2e-4,
        record_points=3,
        checkpoint_every=3,
        lbfgs_refinement_steps=0,
        lbfgs_learning_rate=0.5,
    )


def _config(max_terms: int = 3) -> SlaterSourceSolverConfig:
    stages = (
        AdaptiveDiagonalPathStageConfig(2, 3702, 3703),
        AdaptiveDiagonalPathStageConfig(3, 3704, 3705),
    )
    return SlaterSourceSolverConfig(
        particles=2,
        basis_order=4,
        device="cpu",
        omega=1.0,
        coupling=0.4,
        softening=1.0,
        quadrature_order=24,
        relative_factor_threshold=1e-12,
        factorization_backend="physical",
        source_optimizer=_optimizer(3701),
        stage_optimizer=_optimizer(3703),
        max_terms=max_terms,
        pool_size=4,
        stages=stages[: max_terms - 1],
        overlap_relative_threshold=1e-10,
        condition_threshold=1e8,
        energy_nesting_tolerance=1e-10,
        truth_maximum_dimension=20,
        particle_tensor_maximum_coefficients=100,
    )


def test_canonical_source_and_config_boundaries() -> None:
    config = _config()
    source = canonical_lowest_slater(config)
    expected = torch.zeros((1, 4, 2), dtype=torch.complex128)
    expected[0, :2, :] = torch.eye(2, dtype=torch.complex128)
    torch.testing.assert_close(source, expected)
    with pytest.raises(ValueError, match="greater than source K"):
        replace(config, max_terms=1, stages=()).validate()
    with pytest.raises(ValueError, match="physical-SVD"):
        replace(config, factorization_backend="algebraic").validate()

    registered, record = load_slater_source_command_config(
        Path("docs/experiments/configs/phase37_n4_d6_k4.json")
    )
    assert (registered.particles, registered.basis_order, registered.max_terms) == (
        4,
        6,
        4,
    )
    assert record["checkpoint_path"].endswith("resumed.pt")


def test_clean_command_resume_matches_uninterrupted_and_rejects_changes(
    tmp_path: Path,
) -> None:
    config = _config()
    clean = run_slater_source_adaptive_solver(
        config, checkpoint_path=tmp_path / "clean.pt"
    )
    partial_path = tmp_path / "resumed.pt"
    partial = run_slater_source_adaptive_solver(
        config,
        checkpoint_path=partial_path,
        max_adaptive_stages_this_call=1,
    )
    assert not partial["completed"]
    assert partial["current_terms"] == 2
    checkpoint = load_slater_source_checkpoint(partial_path)
    assert checkpoint["source_completed"]
    assert checkpoint["current_terms"] == 2

    resumed = run_slater_source_adaptive_solver(
        config, checkpoint_path=partial_path, resume=True
    )
    validate_slater_source_result(resumed, require_completed=True)
    assert resumed["current_terms"] == 3
    assert not resumed["source_construction"]["historical_checkpoint_used"]
    assert not resumed["source_construction"]["ci_initializer_used"]
    clean_candidates = [
        stage["selected_candidate"] for stage in clean["stages"][1:]
    ]
    resumed_candidates = [
        stage["selected_candidate"] for stage in resumed["stages"][1:]
    ]
    assert resumed_candidates == clean_candidates
    for clean_stage, resumed_stage in zip(
        clean["stages"], resumed["stages"], strict=True
    ):
        assert (
            abs(
                clean_stage["optimizer_result"]["energy"]
                - resumed_stage["optimizer_result"]["energy"]
            )
            < 1e-11
        )
        point = resumed_stage["optimizer_result"]
        assert point["materialized_antisymmetry_residual"] < 1e-12
        assert point["structural_antisymmetry_residual"] == 0.0
        assert point["structural_counts"]["enumerated_virtual_paths"] == 0
        assert point["structural_counts"]["materialized_particle_coefficients"] == 0

    with pytest.raises(ValueError, match="configuration does not match"):
        run_slater_source_adaptive_solver(
            replace(config, coupling=0.41),
            checkpoint_path=partial_path,
            resume=True,
        )


def test_final_command_state_gradient_matches_materialized_exterior(
    tmp_path: Path,
) -> None:
    config = _config(max_terms=2)
    path = tmp_path / "gradient.pt"
    result = run_slater_source_adaptive_solver(config, checkpoint_path=path)
    adaptive = load_adaptive_diagonal_path_checkpoint(
        path.with_name("gradient.adaptive.pt")
    )
    selected = adaptive["current_orbitals"]
    one_body = harmonic_pair_hamiltonian(
        config.basis_order,
        kappa=0.0,
        omega=config.omega,
        dtype=torch.complex128,
        device="cpu",
    )[0]
    interaction, _ = soft_coulomb_operator(
        config.basis_order,
        quadrature_order=config.quadrature_order,
        coupling=config.coupling,
        softening=config.softening,
        relative_threshold=config.relative_factor_threshold,
        factorization_backend=config.factorization_backend,
        dtype=torch.complex128,
        device="cpu",
    )
    truth_hamiltonian = antisymmetric_many_body_hamiltonian(
        one_body, config.particles, interaction
    )

    polynomial_orbitals = selected.clone().requires_grad_(True)
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        polynomial_orbitals,
        one_body,
        two_body_left=interaction.left,
        two_body_right=interaction.right,
        two_body_weights=interaction.weights,
    )
    polynomial_energy = solve_generalized_hermitian(hamiltonian, overlap).energy
    polynomial_gradient = torch.autograd.grad(
        polynomial_energy, polynomial_orbitals
    )[0]

    explicit_orbitals = selected.clone().requires_grad_(True)
    columns = torch.stack(
        [
            diagonal_path_exterior_coefficients(
                explicit_orbitals[index : index + 1],
                torch.ones(1, dtype=torch.complex128),
            )
            for index in range(explicit_orbitals.shape[0])
        ],
        dim=1,
    )
    explicit = solve_generalized_hermitian(
        columns.mH @ truth_hamiltonian @ columns,
        columns.mH @ columns,
    )
    explicit_gradient = torch.autograd.grad(explicit.energy, explicit_orbitals)[0]

    assert result["completed"]
    torch.testing.assert_close(polynomial_energy, explicit.energy, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        polynomial_gradient, explicit_gradient, atol=1e-8, rtol=1e-8
    )


def test_command_resumes_an_interrupted_source_optimizer(tmp_path: Path) -> None:
    config = _config(max_terms=2)
    path = tmp_path / "source_resume.pt"
    partial = run_slater_source_adaptive_solver(
        config,
        checkpoint_path=path,
        max_source_steps_this_call=1,
    )
    assert not partial["completed"]
    assert not partial["source_result"]["completed"]
    assert partial["adaptive_result"] is None

    resumed = run_slater_source_adaptive_solver(
        config, checkpoint_path=path, resume=True
    )
    validate_slater_source_result(resumed, require_completed=True)
    assert resumed["source_result"]["completed"]
    assert resumed["current_terms"] == 2
