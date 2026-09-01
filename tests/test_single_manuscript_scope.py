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
        "Fixed-bond squared-norm hardness",
        "internal bond is at most two",
        "The structured CHSS boundary is what sharpens the bond from three to two",
        "not needed for the CHSS reduction",
        "No separate \\FEMPS{} method paper is claimed",
        "NOCI-equivalent numerical control",
    ):
        assert phrase in source
    assert "Bond-two exact squared-norm boundary" not in source
    assert "established norm theorem has maximum bond three" not in source


def test_publication_scope_gates_any_future_method_paper() -> None:
    scope = (ROOT / "docs" / "paper" / "SINGLE_MANUSCRIPT_SCOPE.md").read_text(
        encoding="utf-8"
    )
    adr = (
        ROOT / "docs" / "decisions" / "0028-single-manuscript-until-distinctiveness.md"
    ).read_text(encoding="utf-8")
    phase = (
        ROOT / "docs" / "exec-plans" / "active" / "phase40.md"
    ).read_text(
        encoding="utf-8"
    )
    active = (
        ROOT
        / "docs"
        / "exec-plans"
        / "active"
        / "phase41_manuscript_a_theory_closure.md"
    ).read_text(encoding="utf-8")
    completed = (
        ROOT / "docs" / "exec-plans" / "completed" / "phase39.md"
    ).read_text(encoding="utf-8")
    preregistration = (
        ROOT
        / "docs"
        / "decisions"
        / "0030-preregister-n2-explicit-correlation-gate.md"
    ).read_text(encoding="utf-8")

    assert "one publication manuscript" in scope
    assert "frozen internal working note" in scope
    assert "explicit correlation" in scope
    assert "Li--Waintal" in scope
    assert "same-orbital-basis DMRG" in scope
    assert "Maintain one combined structural/no-go manuscript" in adr
    assert "algorithm experiment, not Paper B" in phase
    assert "No title, abstract" in phase
    normalized_phase = " ".join(phase.lower().split())
    assert "external human review" in normalized_phase
    assert "does not create a second manuscript" in normalized_phase
    assert "Keep one manuscript; do not create Paper B" in active
    assert "no further small" in active
    assert "Phase 39 closed the two-paper drift" in completed
    assert "does not authorize a second manuscript" in preregistration
