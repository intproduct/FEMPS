from pathlib import Path

from scripts.verify_phase33_vectorized_transitions import verify_artifact


def test_phase33_vectorized_transitions_artifact() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase33_vectorized_transitions.json")
    )
    assert result["verified"] is True
    assert result["cpu_reference_speedup"] > 10.0
    assert result["blackwell_reference_speedup"] > 5.0
    assert result["selected_backend"] == "cpu"
    assert result["blackwell_admitted"] is True
