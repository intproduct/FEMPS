from pathlib import Path

from scripts.verify_phase30_method_figures import verify_provenance


def test_phase30_method_figures_have_manifest_provenance() -> None:
    result = verify_provenance(
        Path("docs/paper/figures/phase30-method-figure-provenance.json")
    )
    assert result["verified"]
    assert result["figures"] == 4
