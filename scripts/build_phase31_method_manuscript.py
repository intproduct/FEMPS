"""Build the Phase 31 restricted-method manuscript without latexmk/Perl."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/paper/femps_method_manuscript.tex"
BUILD = ROOT / "docs/paper/build_method"
OUTPUT = ROOT / "docs/paper/femps_method_manuscript.pdf"
PROVENANCE = ROOT / "docs/paper/femps_method_manuscript-provenance.json"
MANIFEST = ROOT / "docs/experiments/results/phase30_reproduction_manifest.json"
FIGURE_PROVENANCE = (
    ROOT / "docs/paper/figures/phase30-method-figure-provenance.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(3):
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            return
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def build_manuscript() -> Path:
    BUILD.mkdir(parents=True, exist_ok=True)
    latex = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={BUILD.relative_to(ROOT)}",
        str(SOURCE.relative_to(ROOT)),
    ]
    _run(latex)
    _run(["bibtex", str((BUILD / SOURCE.stem).relative_to(ROOT))])
    _run(latex)
    _run(latex)

    log = (BUILD / f"{SOURCE.stem}.log").read_text(
        encoding="utf-8", errors="replace"
    )
    forbidden = (
        "LaTeX Warning: There were undefined references",
        "LaTeX Warning: Citation `",
        "! LaTeX Error:",
    )
    for marker in forbidden:
        if marker in log:
            raise RuntimeError(f"manuscript build log contains: {marker}")
    shutil.copy2(BUILD / f"{SOURCE.stem}.pdf", OUTPUT)
    provenance = {
        "schema_version": 1,
        "artifact": str(OUTPUT.relative_to(ROOT)),
        "artifact_sha256": _sha256(OUTPUT),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha256(SOURCE),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": _sha256(MANIFEST),
        "figure_provenance": str(FIGURE_PROVENANCE.relative_to(ROOT)),
        "figure_provenance_sha256": _sha256(FIGURE_PROVENANCE),
        "build_command": "python scripts/build_phase31_method_manuscript.py",
        "scientific_boundary": (
            "restricted nonbranching FEMPS method manuscript; all floating-point "
            "benchmark claims remain numerical evidence"
        ),
    }
    PROVENANCE.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return OUTPUT


if __name__ == "__main__":
    print(build_manuscript().relative_to(ROOT))
