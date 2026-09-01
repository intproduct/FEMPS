import hashlib
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


def test_submission_source_hygiene_and_scientific_boundaries() -> None:
    source = (ROOT / "math" / "femps_no_go_manuscript.tex").read_text(
        encoding="utf-8"
    )
    lowered = source.lower()
    for forbidden in (
        "phase 39",
        "phase 40",
        "phase 41",
        "gate pass",
        "gate fail",
        "internal evidence",
        "placeholder",
        "\\includegraphics",
        "todo",
        "tbd",
    ):
        assert forbidden not in lowered
    for required in (
        "row-ordered Cayley determinant",
        "For scalar entries ($\\chi=1$),",
        "The counterexample is only two Slaters",
        "standard \\emph{unnormalized} rational",
        "neither this conjecture nor",
        "unsuitable for\nMonte Carlo or controlled approximation",
        "$D=12$, $K=4$ calculation occupies the $\\binom{12}{6}=924$",
        "preoptimized $D=10$, $K=4$ state",
        "bounded NOCI-equivalent numerical exercise",
        "OpenAI Codex (GPT-5 family",
        "The AI system is not an author",
    ):
        assert required in source


def test_active_plan_index_has_one_operational_plan_and_frozen_exception() -> None:
    active_dir = ROOT / "docs" / "exec-plans" / "active"
    assert {path.name for path in active_dir.glob("*.md")} == {
        "README.md",
        "phase40.md",
        "phase46_external_review_handoff.md",
    }
    phase40 = active_dir / "phase40.md"
    normalized = phase40.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert hashlib.sha256(normalized.encode("utf-8")).hexdigest() == (
        "281aa3deb3c7ecbe3449dfdeba21543d971ebe42c4b8c603b6f1b26d8defbcf6"
    )
    guide = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "active/phase46_external_review_handoff.md" in guide
    assert "active/phase41_manuscript_a_theory_closure.md" not in guide


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
    completed_theory = (
        ROOT
        / "docs"
        / "exec-plans"
        / "completed"
        / "phase41_manuscript_a_theory_closure.md"
    ).read_text(encoding="utf-8")
    active_index = (
        ROOT / "docs" / "exec-plans" / "active" / "README.md"
    ).read_text(encoding="utf-8")
    active = (
        ROOT
        / "docs"
        / "exec-plans"
        / "active"
        / "phase46_external_review_handoff.md"
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
    assert "Keep one manuscript; do not create Paper B" in completed_theory
    assert "no further small" in completed_theory
    assert "sole operational research plan" in active_index
    assert "immutable preregistration" in active_index
    assert "Keep the single combined manuscript" in active
    assert "Phase 39 closed the two-paper drift" in completed
    assert "does not authorize a second manuscript" in preregistration
