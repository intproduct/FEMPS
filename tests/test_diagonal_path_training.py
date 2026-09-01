from pathlib import Path

import torch

from femps.algorithms import (
    DiagonalPathConfig,
    canonical_slater_orbitals,
    run_diagonal_path_variable_projection,
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
    assert not partial["completed"]
    resumed = run_diagonal_path_variable_projection(
        config, checkpoint_path=checkpoint, resume=True
    )
    uninterrupted = run_diagonal_path_variable_projection(config)
    assert resumed["completed"]
    assert abs(resumed["energy"] - uninterrupted["energy"]) < 2e-11
    assert resumed["materialized_antisymmetry_residual"] < 1e-14
