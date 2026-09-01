import json
from pathlib import Path

from scripts.verify_phase28_e4_closure import verify_artifact


def test_recorded_phase28_e4_artifact_passes_independent_verifier() -> None:
    artifact = Path("docs/experiments/results/phase28_e4_closure.json")
    result = verify_artifact(json.loads(artifact.read_text(encoding="utf-8")))
    assert result["verified"]
    assert result["D6_runs"] >= 3
    assert result["D7_runs"] >= 3
