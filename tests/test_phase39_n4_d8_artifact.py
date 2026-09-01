from pathlib import Path

from scripts.verify_phase39_n4_d8_clean_source import verify_artifact


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "experiments" / "results" / "phase39_n4_d8_clean_source.json"


def test_restored_phase39_n4_d8_failure_reconstructs_exactly() -> None:
    result = verify_artifact(ARTIFACT)
    assert result["verified"] is True
    assert result["accepted"] is False
    assert result["optimizer_failure_count"] == 0
    assert result["maximum_clean_resume_energy_difference"] == 0.0
    assert result["maximum_final_error_vs_CI"] > 1e-6
    assert result["maximum_final_variance"] > 1e-5
    assert result["selected_candidates"]["resumed"] == [0, 2, 29]
