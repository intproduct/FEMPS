"""Build and audit the sole combined FEMPS submission manuscript.

The driver deliberately invokes pdflatex and bibtex directly so that a Perl
installation (required by latexmk on some Windows setups) is not part of the
reproducibility contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "math" / "femps_no_go_manuscript.tex"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "pdf" / "combined-manuscript"
FORBIDDEN_LOG_PATTERNS = (
    r"LaTeX Warning:",
    r"Package .* Warning:",
    r"Citation `.*' .* undefined",
    r"Reference `.*' .* undefined",
    r"There were undefined references",
    r"multiply defined",
    r"Overfull \\hbox",
    r"Underfull \\hbox",
    r"Overfull \\vbox",
    r"Underfull \\vbox",
)


def audit_log(log_text: str) -> list[str]:
    """Return every forbidden final-pass diagnostic present in a TeX log."""

    return [
        pattern
        for pattern in FORBIDDEN_LOG_PATTERNS
        if re.search(pattern, log_text, flags=re.IGNORECASE)
    ]


def _executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"{name} was not found; install a TeX distribution with pdflatex "
            "and bibtex on PATH"
        )
    return path


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        transcript = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            f"{' '.join(command)}\n{transcript}"
        )


def _clean_job_files(output_dir: Path, stem: str) -> None:
    for suffix in (
        ".aux",
        ".bbl",
        ".blg",
        ".log",
        ".out",
        ".pdf",
        ".toc",
    ):
        path = output_dir / f"{stem}{suffix}"
        if path.is_file():
            path.unlink()


def build(
    source: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError("manuscript source must remain inside the repository")
    if not output_dir.is_relative_to(ROOT):
        raise ValueError(
            "manuscript output directory must remain inside the repository"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _clean_job_files(output_dir, source.stem)
    pdflatex = _executable("pdflatex")
    bibtex = _executable("bibtex")
    latex_command = [
        pdflatex,
        "--interaction=nonstopmode",
        "--halt-on-error",
        f"--output-directory={output_dir}",
        str(source.relative_to(ROOT)),
    ]
    _run(latex_command)
    _run([bibtex, str((output_dir / source.stem).relative_to(ROOT))])
    _run(latex_command)
    _run(latex_command)

    log_path = output_dir / f"{source.stem}.log"
    pdf_path = output_dir / f"{source.stem}.pdf"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    diagnostics = audit_log(log_text)
    if diagnostics:
        raise RuntimeError(
            "final manuscript log contains forbidden diagnostics: "
            + ", ".join(diagnostics)
        )
    pdf_bytes = pdf_path.read_bytes()
    page_match = re.search(r"\((\d+) pages?, \d+ bytes\)\.", log_text)
    if page_match is None:
        raise RuntimeError("could not recover final PDF page count from TeX log")
    return {
        "source": source.relative_to(ROOT).as_posix(),
        "pdf": pdf_path.relative_to(ROOT).as_posix(),
        "pages": int(page_match.group(1)),
        "bytes": len(pdf_bytes),
        "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "undefined_references": False,
        "layout_warnings": False,
        "latexmk_or_perl_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.source, arguments.output_dir), indent=2))


if __name__ == "__main__":
    main()
