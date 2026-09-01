from pathlib import Path

import pytest
import torch

from femps.algorithms import (
    AdaptiveDiagonalPathConfig,
    AdaptiveDiagonalPathStageConfig,
    DiagonalPathConfig,
    canonical_slater_orbitals,
    load_adaptive_diagonal_path_checkpoint,
    run_bounded_adaptive_diagonal_path,
    solve_generalized_hermitian,
    validate_adaptive_diagonal_path_result,
)
from femps.exterior import (
    diagonal_path_exterior_coefficients,
    diagonal_path_hamiltonian_matrices,
)
from femps.hamiltonians import (
    FactorizedTwoBodyOperator,
    antisymmetric_many_body_hamiltonian,
)


def _complex_random(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(shape, generator=generator, dtype=torch.float64),
        torch.randn(shape, generator=generator, dtype=torch.float64),
    )


def _problem() -> tuple[
    torch.Tensor, torch.Tensor, FactorizedTwoBodyOperator, DiagonalPathConfig
]:
    dimension, particles, terms = 5, 3, 2
    source = canonical_slater_orbitals(
        _complex_random((terms, dimension, particles), 3601)
    )
    raw_one = _complex_random((dimension, dimension), 3602)
    one_body = 0.5 * (raw_one + raw_one.mH)
    raw_left = _complex_random((2, dimension, dimension), 3603)
    raw_right = _complex_random((2, dimension, dimension), 3604)
    interaction = FactorizedTwoBodyOperator(
        0.5 * (raw_left + raw_left.mH),
        0.5 * (raw_right + raw_right.mH),
        torch.tensor([0.08, -0.04], dtype=torch.complex128),
    )
    template = DiagonalPathConfig(
        basis_order=dimension,
        particles=particles,
        terms=terms,
        interaction_model="soft_coulomb",
        steps=3,
        learning_rate=2e-3,
        final_learning_rate=2e-4,
        seed=0,
        record_points=3,
        checkpoint_every=3,
        truth_maximum_dimension=20,
        particle_tensor_maximum_coefficients=1_000,
    )
    return source, one_body, interaction, template


def _schedule(max_terms: int = 4) -> AdaptiveDiagonalPathConfig:
    stages = (
        AdaptiveDiagonalPathStageConfig(3, 3611, 3612),
        AdaptiveDiagonalPathStageConfig(4, 3613, 3614),
    )
    return AdaptiveDiagonalPathConfig(
        max_terms=max_terms,
        pool_size=6,
        stages=stages[: max_terms - 2],
    )


def test_adaptive_schedule_requires_external_cap_and_complete_seeds() -> None:
    with pytest.raises(ValueError, match="greater than source K"):
        AdaptiveDiagonalPathConfig(max_terms=2, pool_size=4, stages=()).validate(2)
    incomplete = AdaptiveDiagonalPathConfig(
        max_terms=4,
        pool_size=4,
        stages=(AdaptiveDiagonalPathStageConfig(3, 1, 2),),
    )
    with pytest.raises(ValueError, match="explicitly cover every K"):
        incomplete.validate(2)


def test_stage_checkpoint_resume_matches_uninterrupted_and_reports_symmetry(
    tmp_path: Path,
) -> None:
    source, one_body, interaction, template = _problem()
    schedule = _schedule()
    uninterrupted = run_bounded_adaptive_diagonal_path(
        source,
        one_body,
        interaction,
        template,
        schedule,
        source_id="test_K2_source",
        operator_id="test_factorized_operator",
        checkpoint_path=tmp_path / "uninterrupted.pt",
    )
    partial_path = tmp_path / "resumed.pt"
    partial = run_bounded_adaptive_diagonal_path(
        source,
        one_body,
        interaction,
        template,
        schedule,
        source_id="test_K2_source",
        operator_id="test_factorized_operator",
        checkpoint_path=partial_path,
        max_stages_this_call=1,
    )
    assert not partial["completed"]
    checkpoint = load_adaptive_diagonal_path_checkpoint(partial_path)
    assert checkpoint["current_terms"] == 3

    resumed = run_bounded_adaptive_diagonal_path(
        source,
        one_body,
        interaction,
        template,
        schedule,
        source_id="test_K2_source",
        operator_id="test_factorized_operator",
        checkpoint_path=partial_path,
        resume=True,
    )
    validate_adaptive_diagonal_path_result(resumed, require_completed=True)
    assert resumed["current_terms"] == 4
    assert resumed["stages_completed_this_call"] == 1
    assert abs(resumed["final_energy"] - uninterrupted["final_energy"]) < 1e-11
    assert [stage["selected_candidate"] for stage in resumed["stages"]] == [
        stage["selected_candidate"] for stage in uninterrupted["stages"]
    ]
    assert resumed["stages"][-1]["optimized_orbitals_sha256"] == uninterrupted[
        "stages"
    ][-1]["optimized_orbitals_sha256"]
    assert resumed["structural_antisymmetry_residual"] == 0.0
    assert resumed["enumerated_virtual_paths"] == 0
    assert resumed["materialized_particle_coefficients"] == 0
    for stage in resumed["stages"]:
        point = stage["optimizer_result"]
        assert stage["source_nesting_max_abs_error"] < 1e-13
        assert point["polynomial_explicit_absolute_difference"] < 1e-10
        assert point["materialized_antisymmetry_residual"] < 1e-12
        assert point["peak_cpu_rss_bytes"] > 0

    changed_source = source.clone()
    changed_source[0, 0, 0] += 1e-8
    with pytest.raises(ValueError, match="source does not match"):
        run_bounded_adaptive_diagonal_path(
            changed_source,
            one_body,
            interaction,
            template,
            schedule,
            source_id="test_K2_source",
            operator_id="test_factorized_operator",
            checkpoint_path=partial_path,
            resume=True,
        )


def test_final_public_state_gradient_matches_materialized_exterior(
    tmp_path: Path,
) -> None:
    source, one_body, interaction, template = _problem()
    result = run_bounded_adaptive_diagonal_path(
        source,
        one_body,
        interaction,
        template,
        _schedule(max_terms=3),
        source_id="gradient_K2_source",
        operator_id="gradient_factorized_operator",
        checkpoint_path=tmp_path / "gradient.pt",
    )
    checkpoint = load_adaptive_diagonal_path_checkpoint(tmp_path / "gradient.pt")
    selected = checkpoint["current_orbitals"]
    truth_hamiltonian = antisymmetric_many_body_hamiltonian(
        one_body, template.particles, interaction
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
    torch.testing.assert_close(
        polynomial_energy, explicit.energy, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        polynomial_gradient, explicit_gradient, atol=1e-8, rtol=1e-8
    )
