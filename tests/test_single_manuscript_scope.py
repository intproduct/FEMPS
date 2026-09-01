from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_combined_manuscript_keeps_structural_results_and_chi2_boundary() -> None:
    source = (ROOT / "math" / "femps_no_go_manuscript.tex").read_text(
        encoding="utf-8"
    )
    for phrase in (
        "Structural result I: exact particle-TT ranks",
        "Structural result II: universal exchange floor",
        "Structural result III: known flat Slater particle spectrum",
        "Bond-two exact squared-norm boundary",
        "maximum bond $\\chi=2$",
        "established norm theorem has maximum bond three",
        "No separate \\FEMPS{} method paper is claimed",
        "NOCI-equivalent numerical control",
    ):
        assert phrase in source


def test_publication_scope_gates_any_future_method_paper() -> None:
    scope = (ROOT / "docs" / "paper" / "SINGLE_MANUSCRIPT_SCOPE.md").read_text(
        encoding="utf-8"
    )
    adr = (
        ROOT / "docs" / "decisions" / "0028-single-manuscript-until-distinctiveness.md"
    ).read_text(encoding="utf-8")
    phase = (ROOT / "docs" / "exec-plans" / "active" / "phase39.md").read_text(
        encoding="utf-8"
    )

    assert "one publication manuscript" in scope
    assert "frozen internal working note" in scope
    assert "explicit correlation" in scope
    assert "Li--Waintal" in scope
    assert "same-orbital-basis DMRG" in scope
    assert "Maintain one combined structural/no-go manuscript" in adr
    assert "NOCI-equivalent numerical exercise" in phase
    assert "No second method manuscript is opened" in phase
