from pathlib import Path

from scripts.verify_phase28_soft_coulomb_basis_extension import verify_artifact


def test_recorded_soft_coulomb_basis_extension_passes() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase28_soft_coulomb_basis_extension.json")
    )
    assert result["verified"]
    assert result["dimensions"] == [8, 10, 12]
