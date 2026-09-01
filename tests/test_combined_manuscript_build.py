from scripts.build_combined_manuscript import audit_log


def test_final_tex_log_audit_accepts_clean_log() -> None:
    assert audit_log("Output written on manuscript.pdf (15 pages, 100 bytes).") == []


def test_final_tex_log_audit_rejects_references_and_layout_warnings() -> None:
    log = "\n".join(
        (
            "LaTeX Warning: There were undefined references.",
            "Overfull \\hbox (1.0pt too wide)",
        )
    )
    diagnostics = audit_log(log)
    assert r"LaTeX Warning:" in diagnostics
    assert r"There were undefined references" in diagnostics
    assert r"Overfull \\hbox" in diagnostics
