import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PRA_SOURCE = ROOT / "paper" / "femps_pra_manuscript.tex"
AUDIT_SOURCE = ROOT / "paper" / "femps_pra_evidence_audit.tex"
FORMALIZATION_AUDIT = ROOT / "PROOF_FORMALIZATION_AUDIT.md"


def test_pra_source_uses_revtex_and_preserves_the_scientific_core() -> None:
    text = PRA_SOURCE.read_text(encoding="utf-8")
    required = (
        r"\documentclass[aps,pra,reprint",
        "Structural result I: exact particle-TT ranks",
        "Structural result II: universal exchange floor",
        "Structural result III: known flat Slater particle spectrum",
        "Fixed-bond squared-norm hardness",
        "Exact rational-polynomial point evaluation",
        r"$25.0494711446$",
        "preoptimized $D=10$, $K=4$ state",
        "NOCI-equivalent numerical exercise",
        "structural antisymmetry residual is exactly zero",
    )
    for phrase in required:
        assert phrase in text


def test_publication_source_contains_no_internal_scaffolding() -> None:
    text = PRA_SOURCE.read_text(encoding="utf-8").lower()
    forbidden = (
        "sha-256",
        "python -m pytest",
        "repository root",
        "exact-certificate hash",
        "reproducibility checks",
        "phase 39",
        "gate g",
        "placeholder",
    )
    for phrase in forbidden:
        assert phrase not in text


def test_human_audit_is_self_contained_and_does_not_fake_signoff() -> None:
    text = AUDIT_SOURCE.read_text(encoding="utf-8")
    required = (
        "E1--E3: exchange rank in particle factorization",
        "E4: fixed-bond exact squared-norm obstruction",
        "E5: exact point evaluation in a rational Legendre basis",
        "E9: selected interacting numerical calculation",
        "Determinant transition algorithm",
        "Optimization and initialization lineage",
        "Results and resource account",
        "Reviewer decision:",
        "Final human sign-off",
    )
    for phrase in required:
        assert phrase in text
    assert "intentionally not prefilled" in text
    assert "SHA-256" not in text


def test_proof_formalization_records_flags_and_no_claim_change() -> None:
    manuscript = PRA_SOURCE.read_text(encoding="utf-8")
    audit = FORMALIZATION_AUDIT.read_text(encoding="utf-8")
    statement_pattern = re.compile(
        r"\\begin\{(theorem|lemma|corollary)\}(.*?)\\end\{\1\}", re.DOTALL
    )
    statements = statement_pattern.findall(manuscript)
    payload = "\n".join(kind + body for kind, body in statements).encode()
    assert len(statements) == 14
    assert hashlib.sha256(payload).hexdigest() == (
        "f570e133b0bb22d306c13915fe296cc8606e7022c2b5b32566649665d09d3e03"
    )
    assert manuscript.count(r"\begin{auditflag}") == 4
    assert manuscript.count(r"\begin{proof}") == 14
    for dependency in ("CHSS", "Valiant", "Meiburg"):
        assert dependency in audit
    assert "Was new reasoning introduced?" in audit
    assert "byte-for-byte identical" in audit
