from pathlib import Path

from scripts.verify_phase30_reproduction_manifest import verify_manifest


def test_phase30_reproduction_manifest_verifies_all_admitted_artifacts() -> None:
    result = verify_manifest(
        Path("docs/experiments/results/phase30_reproduction_manifest.json")
    )
    assert result["verified"]
    assert result["entries"] == 12
