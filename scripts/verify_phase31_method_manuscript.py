"""Verify the restricted FEMPS method manuscript against admitted evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST = Path("docs/experiments/results/phase30_reproduction_manifest.json")
FIGURE_PROVENANCE = Path(
    "docs/paper/figures/phase30-method-figure-provenance.json"
)
MANUSCRIPT_PROVENANCE = Path(
    "docs/paper/femps_method_manuscript-provenance.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manuscript(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    provenance = json.loads(FIGURE_PROVENANCE.read_text(encoding="utf-8"))
    claim_ids = {entry["id"] for entry in manifest["entries"]}

    required_scope = (
        "Restricted Nonbranching Functional Exterior Matrix Product States",
        "first quantization",
        "continuous functional bases",
        "not an efficient generic",
        "numerical evidence",
        "antisymmetry residual",
        "Direct CI remains faster",
        "CPU remains faster",
        "batched $K^2$ determinant transitions",
        "no $N=8$",
    )
    for phrase in required_scope:
        if phrase not in text:
            raise AssertionError(f"missing manuscript scope boundary: {phrase}")

    forbidden_claims = (
        "generic FEMPS is efficiently contractible",
        "FEMPS outperforms CI",
        "FEMPS outperforms DMRG",
        "proof of asymptotic scaling",
    )
    for phrase in forbidden_claims:
        if phrase.lower() in text.lower():
            raise AssertionError(f"forbidden broad method claim: {phrase}")

    missing = sorted(
        claim_id
        for claim_id in claim_ids
        if f"\\claimid{{{claim_id}}}" not in text
    )
    if missing:
        raise AssertionError(f"manifest claims absent from manuscript: {missing}")

    mapped_floats = {
        "tab:n4-k": {"n4_soft_coulomb_transferability"},
        "tab:n6-seeds": {"n6_soft_coulomb_multiseed"},
        "tab:n6-dk": {"n6_independent_dk_convergence"},
        "tab:cost-counts": {"matched_n4_n6_transition_cost"},
        "tab:vectorized-backend": {"n6_vectorized_transition_backend"},
        "fig:convergence": {
            "n4_soft_coulomb_transferability",
            "n4_soft_coulomb_basis_extension",
            "n4_soft_coulomb_high_basis_correlation",
            "n6_soft_coulomb_pilot",
            "n6_soft_coulomb_multiseed",
        },
        "fig:structure-cost": {"matched_n4_n6_transition_cost"},
    }
    for label, identifiers in mapped_floats.items():
        label_position = text.find(f"\\label{{{label}}}")
        if label_position < 0:
            raise AssertionError(f"missing registered manuscript float: {label}")
        float_start = max(
            text.rfind("\\begin{table}", 0, label_position),
            text.rfind("\\begin{figure}", 0, label_position),
        )
        table_end = text.find("\\end{table}", label_position)
        figure_end = text.find("\\end{figure}", label_position)
        candidates = [end for end in (table_end, figure_end) if end >= 0]
        if float_start < 0 or not candidates:
            raise AssertionError(f"cannot locate float boundaries: {label}")
        float_block = text[float_start : min(candidates)]
        for identifier in identifiers:
            if f"\\claimid{{{identifier}}}" not in float_block:
                raise AssertionError(
                    f"float {label} is not mapped to manifest claim {identifier}"
                )

    figure_files = {
        "docs/paper/figures/femps-convergence-summary.pdf",
        "docs/paper/figures/femps-structure-cost-summary.pdf",
    }
    provenance_figures = {Path(figure).as_posix() for figure in provenance["figures"]}
    if not figure_files.issubset(provenance_figures):
        raise AssertionError("manuscript figures are absent from provenance")
    for figure in figure_files:
        if Path(figure).name not in text:
            raise AssertionError(f"provenance figure absent from manuscript: {figure}")

    build_script = Path("scripts/build_phase31_method_manuscript.py")
    if not build_script.is_file() or build_script.name not in text:
        raise AssertionError("reproducible manuscript build command is missing")

    manuscript_provenance = json.loads(
        MANUSCRIPT_PROVENANCE.read_text(encoding="utf-8")
    )
    if manuscript_provenance["schema_version"] != 1:
        raise AssertionError("unsupported manuscript provenance schema")
    provenance_targets = {
        "artifact": "artifact_sha256",
        "source": "source_sha256",
        "manifest": "manifest_sha256",
        "figure_provenance": "figure_provenance_sha256",
    }
    for path_key, hash_key in provenance_targets.items():
        target = Path(manuscript_provenance[path_key])
        if not target.is_file() or _sha256(target) != manuscript_provenance[hash_key]:
            raise AssertionError(f"stale manuscript provenance target: {path_key}")

    return {
        "verified": True,
        "manifest_claims": len(claim_ids),
        "mapped_numerical_floats": len(mapped_floats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manuscript",
        nargs="?",
        type=Path,
        default=Path("docs/paper/femps_method_manuscript.tex"),
    )
    args = parser.parse_args()
    print(json.dumps(verify_manuscript(args.manuscript), indent=2))


if __name__ == "__main__":
    main()
