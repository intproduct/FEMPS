from pathlib import Path

from scripts.verify_phase44_n4_explicit_correlation_d_gate import verify


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
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


def test_phase44_failed_gate_and_internal_advantage_verify() -> None:
    result = verify(ARTIFACT, MANIFEST)
    assert result["verified"] is True
    assert result["phase44_interacting_d_gate_pass"] is False
    assert result["two_consecutive_D_advantage_pass"] is True
    assert result["consecutive_advantage_pairs"] == [[4, 6]]
    assert result["all_confirmation_gates_pass"] is True
    assert result["maximum_observable_difference"] <= 3e-14
    assert result["verified_checkpoint_count"] == 7
    assert result["external_independent_replication_complete"] is False
    assert result["paper_b_authorized"] is False
