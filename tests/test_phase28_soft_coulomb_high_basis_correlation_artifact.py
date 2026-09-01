from pathlib import Path

from scripts.verify_phase28_soft_coulomb_high_basis_correlation import verify_artifact


def test_recorded_high_basis_correlation_audit_passes() -> None:
    result = verify_artifact(
        Path(
            "docs/experiments/results/"
            "phase28_soft_coulomb_high_basis_correlation.json"
        )
    )
    assert result["verified"]
    assert result["audit_pass"]
