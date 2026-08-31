from pathlib import Path
import uuid

import pytest
import torch

from femps.algorithms import (
    FiniteAgpConfig,
    canonical_pair_matrices,
    run_finite_agp_variable_projection,
)


def test_canonical_pair_map_fixes_skew_norm_and_anchor_phase() -> None:
    generator = torch.Generator().manual_seed(701)
    raw_real = torch.randn(3, 6, 6, generator=generator, dtype=torch.float64)
    raw_imaginary = torch.randn(
        3, 6, 6, generator=generator, dtype=torch.float64
    )
    pair_matrices = canonical_pair_matrices(
        torch.complex(raw_real, raw_imaginary)
    )
    torch.testing.assert_close(
        pair_matrices + pair_matrices.transpose(1, 2),
        torch.zeros_like(pair_matrices),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(pair_matrices, dim=(1, 2)),
        torch.ones(3, dtype=torch.float64),
    )
    upper = torch.triu_indices(6, 6, offset=1)
    values = pair_matrices[:, upper[0], upper[1]]
    anchors = torch.argmax(torch.abs(values), dim=1)
    anchor_values = values[torch.arange(3), anchors]
    assert torch.max(torch.abs(anchor_values.imag)) < 2e-16
    assert torch.all(anchor_values.real > 0)


def test_finite_agp_variable_projection_resume_matches_uninterrupted() -> None:
    config = FiniteAgpConfig(
        basis_order=5,
        particles=4,
        agp_terms=2,
        kappa=0.1,
        steps=8,
        learning_rate=3e-3,
        final_learning_rate=1e-4,
        seed=4,
        device="cpu",
        record_points=4,
        checkpoint_every=2,
    )
    uninterrupted = run_finite_agp_variable_projection(config)
    checkpoint = Path("tmp") / f"finite-agp-{uuid.uuid4().hex}.pt"
    try:
        partial = run_finite_agp_variable_projection(
            config,
            checkpoint_path=checkpoint,
            max_steps_this_call=4,
        )
        assert not partial["completed"]
        resumed = run_finite_agp_variable_projection(
            config,
            checkpoint_path=checkpoint,
            resume=True,
        )
        assert resumed["completed"]
        assert resumed["resumed"]
        assert resumed["final_energy"] == pytest.approx(
            uninterrupted["final_energy"], abs=3e-11
        )
        assert resumed["error_vs_finite_basis"] >= -1e-10
        assert resumed["final_retained_rank"] >= 1
        assert resumed["polynomial_explicit_absolute_difference"] < 2e-9
    finally:
        checkpoint.unlink(missing_ok=True)


def test_finite_agp_config_rejects_odd_particle_count() -> None:
    with pytest.raises(ValueError, match="even particle"):
        FiniteAgpConfig(particles=3).validate()
