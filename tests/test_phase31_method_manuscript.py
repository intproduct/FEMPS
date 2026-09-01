from pathlib import Path

from scripts.verify_phase31_method_manuscript import verify_manuscript


def test_frozen_restricted_method_note_maps_its_admitted_evidence() -> None:
    result = verify_manuscript(Path("docs/paper/femps_method_manuscript.tex"))
    assert result["verified"]
    assert result["frozen_manifest_claims"] == 14
    assert result["post_freeze_claims"] == ["n4_clean_source_seed_robustness"]
    assert result["mapped_numerical_floats"] == 9
    assert set(result["historical_provenance_snapshot_matches"]) == {
        "artifact",
        "source",
        "manifest",
        "figure_provenance",
    }
