import json
from pathlib import Path

from scripts import compare_phase44_external_reproduction as comparison


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = (
    ROOT
    / "docs"
    / "experiments"
    / "results"
    / "phase44_n4_explicit_correlation_d_gate.json"
)
MANIFEST = (
    ROOT
    / "docs"
    / "experiments"
    / "results"
    / "phase44_optimizer_checkpoint_manifest.json"
)


def _verified_external_mode(*_args, **_kwargs):
    return {
        "verified": True,
        "checkpoint_verification_mode": "artifact_self_contained",
    }


def test_identical_numerics_do_not_self_attest_external_independence(
    monkeypatch,
) -> None:
    monkeypatch.setattr(comparison, "verify", _verified_external_mode)
    result = comparison.compare(PRIMARY, PRIMARY, MANIFEST)
    assert result["numerical_reproduction_pass"] is True
    assert result["decisions_identical"] is True
    assert all(record["pass"] for record in result["energy_comparisons"])
    assert result["external_independent_replication_complete"] is False
    assert result["external_independence_requires_named_human_attestation"] is True
    assert result["paper_b_authorized"] is False


def test_changed_frozen_axis_fails_before_numerical_verification(
    monkeypatch, tmp_path: Path
) -> None:
    artifact = json.loads(PRIMARY.read_text(encoding="utf-8"))
    artifact["frozen_axes"]["D"] = [4, 6]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(artifact), encoding="utf-8")

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("verification should not run after identity mismatch")

    monkeypatch.setattr(comparison, "verify", should_not_run)
    result = comparison.compare(changed, PRIMARY, MANIFEST)
    assert result["identity_matches"]["frozen_axes"] is False
    assert result["reproduction_self_verified"] is False
    assert result["numerical_reproduction_pass"] is False


def test_internally_inconsistent_reproduction_is_reported(monkeypatch) -> None:
    def reject(*_args, **_kwargs):
        raise AssertionError("raw archive hash mismatch")

    monkeypatch.setattr(comparison, "verify", reject)
    result = comparison.compare(PRIMARY, PRIMARY, MANIFEST)
    assert result["reproduction_self_verified"] is False
    assert "raw archive hash mismatch" in result["verification_error"]
    assert result["numerical_reproduction_pass"] is False
