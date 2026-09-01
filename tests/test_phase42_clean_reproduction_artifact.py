from pathlib import Path

from scripts.verify_phase42_clean_reproduction import verify


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "experiments"
    / "results"
    / "phase42_phase40_clean_reproduction.json"
)


def test_phase42_clean_reproduction_artifact_verifies() -> None:
    result = verify(ARTIFACT)
    assert result["verified"] is True
    assert result["clean_source_reproduction_pass"] is True
    assert result["maximum_correlated_observable_difference"] == 0.0
    assert result["maximum_noci_observable_difference"] == 0.0
    assert result["consecutive_advantage_pairs"] == [[2, 4], [4, 6], [6, 8]]
    assert result["external_independent_replication_complete"] is False
