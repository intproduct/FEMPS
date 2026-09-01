from pathlib import Path

from scripts.verify_phase29_n6_soft_coulomb_pilot import verify_artifact


def test_recorded_n6_soft_coulomb_pilot_passes() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase29_n6_soft_coulomb_pilot.json")
    )
    assert result["verified"]
    assert result["pilot_pass"]
