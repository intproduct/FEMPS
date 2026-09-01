from pathlib import Path

from scripts.verify_phase31_method_manuscript import verify_manuscript


def test_restricted_method_manuscript_maps_all_admitted_evidence() -> None:
    result = verify_manuscript(Path("docs/paper/femps_method_manuscript.tex"))
    assert result["verified"]
    assert result["manifest_claims"] == 13
    assert result["mapped_numerical_floats"] == 9
