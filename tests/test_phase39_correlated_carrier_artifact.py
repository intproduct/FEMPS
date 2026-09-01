from pathlib import Path

from scripts.verify_phase39_correlated_carrier_prototype import verify


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "experiments"
    / "results"
    / "phase39_correlated_carrier_prototype.json"
)


def test_phase39_correlated_carrier_artifact_reconstructs() -> None:
    result = verify(ARTIFACT)
    assert result["verified"] is True
    assert len(result["projection_axis"]) == 5
    assert len(result["ci_axis"]) == 6
