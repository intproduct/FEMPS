from pathlib import Path

from scripts.verify_phase29_n4_n6_matched_cost import verify_artifact


def test_recorded_n4_n6_matched_cost_audit_passes() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase29_n4_n6_matched_cost.json")
    )
    assert result["verified"]
    assert result["matched_cost_audit_pass"]
