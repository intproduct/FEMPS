"""Build and audit a REVTeX manuscript with MiKTeX on Windows.

REVTeX writes a job-local ``*Notes.bib`` file beside the auxiliary file.  The
ordinary manuscript builder runs BibTeX from the repository root, which makes
that generated database invisible when ``--output-directory`` is used.  This
driver runs BibTeX inside the output directory while preserving the repository
root as a bibliography search path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from build_combined_manuscript import ROOT, audit_log


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"{name} is required to build the manuscript")
    return executable


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        transcript = (completed.stdout + "\n" + completed.stderr).strip()
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n{transcript}"
        )


def build(source: Path, output_dir: Path) -> dict[str, object]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not source.is_relative_to(ROOT) or not output_dir.is_relative_to(ROOT):
        raise ValueError("source and output directory must remain in the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    for path in output_dir.glob(f"{stem}*"):
        if path.is_file() and path.suffix in {
            ".aux", ".bbl", ".bib", ".blg", ".log", ".out", ".pdf", ".toc"
        }:
            path.unlink()

    latex = [
        _executable("pdflatex"),
        "--interaction=nonstopmode",
        "--halt-on-error",
        f"--output-directory={output_dir}",
        str(source.relative_to(ROOT)),
    ]
    _run(latex, cwd=ROOT)

    bib_env = os.environ.copy()
    inherited = bib_env.get("BIBINPUTS", "")
    bib_env["BIBINPUTS"] = os.pathsep.join(
        part for part in (str(ROOT), inherited) if part
    )
    _run([_executable("bibtex"), stem], cwd=output_dir, env=bib_env)
    _run(latex, cwd=ROOT)
    _run(latex, cwd=ROOT)

    log_path = output_dir / f"{stem}.log"
    pdf_path = output_dir / f"{stem}.pdf"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    diagnostics = [
        diagnostic
        for diagnostic in audit_log(log_text)
        if diagnostic not in {r"Underfull \\hbox", r"Underfull \\vbox"}
    ]
    if diagnostics:
        raise RuntimeError(
            "final manuscript log contains forbidden diagnostics: "
            + ", ".join(diagnostics)
        )
    page_match = re.search(r"\((\d+) pages?, \d+ bytes\)\.", log_text)
    if page_match is None:
        raise RuntimeError("could not determine the final PDF page count")
    return {
        "source": source.relative_to(ROOT).as_posix(),
        "pdf": pdf_path.relative_to(ROOT).as_posix(),
        "pages": int(page_match.group(1)),
        "bytes": pdf_path.stat().st_size,
        "undefined_references": False,
        "layout_warnings": False,
        "underfull_boxes": len(re.findall(r"Underfull \\[hv]box", log_text)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
