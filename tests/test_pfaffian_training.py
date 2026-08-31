from pathlib import Path
import uuid

import pytest

from femps.algorithms import PfaffianPairConfig, run_pfaffian_pair


def test_checkpoint_resume_matches_uninterrupted_training() -> None:
    config = PfaffianPairConfig(
        basis_order=4,
        kappa=0.0,
        steps=30,
        learning_rate=2e-2,
        final_learning_rate=1e-4,
        seed=9,
        device="cpu",
        record_points=6,
        checkpoint_every=5,
    )
    uninterrupted = run_pfaffian_pair(config)
    checkpoint = Path("tmp") / f"pfaffian-training-{uuid.uuid4().hex}.pt"
    try:
        partial = run_pfaffian_pair(
            config,
            checkpoint_path=checkpoint,
            max_steps_this_call=12,
        )
        assert not partial["completed"]
        resumed = run_pfaffian_pair(
            config,
            checkpoint_path=checkpoint,
            resume=True,
        )
        assert resumed["completed"]
        assert resumed["resumed"]
        assert resumed["final_energy"] == pytest.approx(
            uninterrupted["final_energy"], abs=2e-13
        )
    finally:
        checkpoint.unlink(missing_ok=True)
