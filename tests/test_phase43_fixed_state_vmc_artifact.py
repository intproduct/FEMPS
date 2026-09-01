from pathlib import Path

from scripts.verify_phase43_fixed_state_vmc_validation import verify


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "experiments"
    / "results"
    / "phase43_fixed_state_vmc_validation.json"
)


def test_phase43_fixed_state_vmc_artifact_verifies() -> None:
    result = verify(ARTIFACT)
    assert result["verified"] is True
    assert result["phase43_fixed_state_validation_pass"] is True
    assert result["maximum_observable_difference"] <= 2e-14
    assert result["maximum_gradient_difference"] <= 2e-14
    assert result["external_independent_replication_complete"] is False
    assert result["interacting_n4_complete"] is False
    assert result["paper_b_authorized"] is False
