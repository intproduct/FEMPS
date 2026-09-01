from pathlib import Path

from scripts.verify_phase32_n6_resource_audit import verify_artifact


def test_phase32_n6_resource_audit_artifact() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase32_n6_resource_audit.json")
    )
    assert result["verified"] is True
    assert result["dimensions"] == [8, 10, 12]
