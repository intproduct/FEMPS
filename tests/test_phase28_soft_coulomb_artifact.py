from pathlib import Path

from scripts.verify_phase28_soft_coulomb_transferability import verify_artifact


def test_recorded_soft_coulomb_transferability_artifact_passes() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase28_soft_coulomb_transferability.json")
    )
    assert result["verified"]
    assert result["transferability_pass"]
