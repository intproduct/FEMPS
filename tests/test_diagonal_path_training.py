from pathlib import Path

import pytest
import torch

from femps.algorithms import (
    DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION,
    DIAGONAL_PATH_RESULT_SCHEMA_VERSION,
    DiagonalPathConfig,
    canonical_slater_orbitals,
    embed_diagonal_path_orbitals,
    extend_diagonal_path_terms,
    load_diagonal_path_checkpoint,
    run_diagonal_path_variable_projection,
    validate_diagonal_path_checkpoint,
    validate_diagonal_path_result,
)


def test_canonical_slater_orbitals_are_orthonormal() -> None:
    generator = torch.Generator().manual_seed(2811)
    raw = torch.randn((3, 6, 2), generator=generator, dtype=torch.float64)
    orbitals = canonical_slater_orbitals(raw)
    gram = orbitals.transpose(1, 2) @ orbitals
    torch.testing.assert_close(
        gram,
        torch.eye(2, dtype=torch.float64).expand(3, 2, 2),
        atol=3e-14,
        rtol=3e-14,
    )


def test_nested_basis_embedding_preserves_orbitals_and_orthonormality() -> None:
    source = canonical_slater_orbitals(
        torch.randn((2, 5, 3), generator=torch.Generator().manual_seed(28))
    )
    embedded = embed_diagonal_path_orbitals(source, 8)
    torch.testing.assert_close(embedded[:, :5, :], source)
    assert torch.count_nonzero(embedded[:, 5:, :]) == 0
    torch.testing.assert_close(
        embedded.transpose(1, 2) @ embedded,
        torch.eye(3).expand(2, 3, 3),
    )


def test_term_embedding_preserves_source_and_is_seeded() -> None:
    source = torch.complex(
        torch.randn(
            (2, 6, 4),
            generator=torch.Generator().manual_seed(2808),
            dtype=torch.float64,
        ),
        torch.randn(
            (2, 6, 4),
            generator=torch.Generator().manual_seed(2809),
            dtype=torch.float64,
        ),
    )
    extended = extend_diagonal_path_terms(source, 5, seed=2810)
    repeated = extend_diagonal_path_terms(source, 5, seed=2810)

    assert extended.shape == (5, 6, 4)
    assert torch.equal(extended[:2], source)
    assert torch.equal(extended, repeated)
    torch.testing.assert_close(
        extended[2:].mH @ extended[2:],
        torch.eye(4, dtype=torch.complex128).expand(3, 4, 4),
        atol=3e-14,
        rtol=3e-14,
    )


def test_k_one_noninteracting_training_is_exact_and_audited() -> None:
    config = DiagonalPathConfig(
        basis_order=4,
        particles=2,
        terms=1,
        kappa=0.0,
        steps=2,
        learning_rate=1e-3,
        final_learning_rate=1e-4,
        record_points=2,
        checkpoint_every=1,
    )
    result = run_diagonal_path_variable_projection(config)
    validate_diagonal_path_result(result, require_completed=True)
    assert result["schema_version"] == DIAGONAL_PATH_RESULT_SCHEMA_VERSION
    assert result["completed"]
    assert abs(result["energy"] - 2.0) < 1e-12
    assert result["finite_basis_reference_energy"] == 2.0
    assert abs(result["error_vs_finite_basis"]) < 1e-12
    assert result["polynomial_explicit_absolute_difference"] < 1e-12
    assert result["energy_variance"] < 1e-12
    assert result["norm_error"] < 1e-12
    assert result["structural_antisymmetry_residual"] == 0.0
    assert result["materialized_antisymmetry_residual"] < 1e-14
    assert result["structural_counts"]["enumerated_virtual_paths"] == 0
    assert result["peak_cpu_rss_bytes"] > 0
    assert result["cpu_memory"]["samples"] >= 2
    assert result["total_elapsed_seconds_this_call"] >= 0


def test_optional_lbfgs_refinement_is_audited_and_nonworsening() -> None:
    config = DiagonalPathConfig(
        basis_order=4,
        particles=2,
        terms=2,
        kappa=0.2,
        steps=2,
        learning_rate=2e-3,
        final_learning_rate=2e-4,
        seed=9,
        record_points=2,
        checkpoint_every=2,
        lbfgs_refinement_steps=5,
    )
    result = run_diagonal_path_variable_projection(config)
    refinement = result["refinement"]
    assert refinement["closure_calls"] >= 1
    assert result["energy"] <= refinement["initial_energy"] + 1e-12
    assert result["structural_antisymmetry_residual"] == 0.0


def test_vectorized_soft_coulomb_lbfgs_accepts_contiguous_parameter_gradient() -> None:
    from femps.hamiltonians import harmonic_pair_hamiltonian, soft_coulomb_operator

    dimension = 6
    one_body = harmonic_pair_hamiltonian(
        dimension, kappa=0.0, dtype=torch.complex128, device="cpu"
    )[0]
    interaction = soft_coulomb_operator(
        dimension,
        quadrature_order=32,
        relative_threshold=1e-13,
        factorization_backend="physical",
        dtype=torch.complex128,
        device="cpu",
    )[0]
    config = DiagonalPathConfig(
        basis_order=dimension,
        particles=4,
        terms=4,
        interaction_model="soft_coulomb",
        soft_coulomb_quadrature_order=32,
        steps=2,
        learning_rate=1e-3,
        final_learning_rate=1e-4,
        seed=3304,
        record_points=2,
        checkpoint_every=2,
        lbfgs_refinement_steps=2,
        truth_maximum_dimension=20,
    )
    generator = torch.Generator().manual_seed(3304)
    initial_orbitals = canonical_slater_orbitals(
        torch.complex(
            torch.randn(
                (4, dimension, 4), generator=generator, dtype=torch.float64
            ),
            torch.randn(
                (4, dimension, 4), generator=generator, dtype=torch.float64
            ),
        )
    )
    assert not initial_orbitals.is_contiguous()
    result = run_diagonal_path_variable_projection(
        config,
        initial_orbitals=initial_orbitals,
        operators=(one_body, interaction),
        operator_id="test_vectorized_soft_coulomb_lbfgs",
    )

    assert result["completed"]
    assert result["refinement"]["closure_calls"] >= 1


def test_checkpoint_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    config = DiagonalPathConfig(
        basis_order=4,
        particles=2,
        terms=2,
        kappa=0.2,
        steps=4,
        learning_rate=2e-3,
        final_learning_rate=2e-4,
        seed=12,
        record_points=4,
        checkpoint_every=2,
    )
    checkpoint = tmp_path / "diagonal_path.pt"
    partial = run_diagonal_path_variable_projection(
        config, checkpoint_path=checkpoint, max_steps_this_call=2
    )
    payload = load_diagonal_path_checkpoint(checkpoint)
    assert payload["schema_version"] == DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION
    incompatible = dict(payload)
    incompatible["schema_version"] = 999
    with pytest.raises(ValueError, match="schema version"):
        validate_diagonal_path_checkpoint(incompatible)
    assert not partial["completed"]
    resumed = run_diagonal_path_variable_projection(
        config, checkpoint_path=checkpoint, resume=True
    )
    uninterrupted = run_diagonal_path_variable_projection(config)
    assert resumed["completed"]
    assert abs(resumed["energy"] - uninterrupted["energy"]) < 2e-11
    assert resumed["materialized_antisymmetry_residual"] < 1e-14
