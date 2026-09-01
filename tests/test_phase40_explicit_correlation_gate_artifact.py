from pathlib import Path

from scripts.verify_phase40_explicit_correlation_gate import verify


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "experiments"
    / "results"
    / "phase40_explicit_correlation_gate.json"
)


def test_phase40_explicit_correlation_gate_reconstructs() -> None:
    result = verify(ARTIFACT)
    assert result["verified"] is True
    assert result["phase40_differentiator_pass"] is True
    assert result["correlated_points_reconstructed"] == 72
    assert result["noci_points_reconstructed"] == 54
    assert result["consecutive_advantage_pairs"] == [[2, 4], [4, 6], [6, 8]]
    assert result["independent_reproduction_still_required"] is True
