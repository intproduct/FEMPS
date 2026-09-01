from pathlib import Path

from scripts.verify_phase29_n6_multiseed_stability import verify_artifact


def test_recorded_n6_multiseed_stability_passes() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase29_n6_multiseed_stability.json")
    )
    assert result["verified"]
    assert result["multiseed_pass"]
