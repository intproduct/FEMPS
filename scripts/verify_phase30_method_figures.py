"""Verify Phase 30 paper-figure source and output provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_provenance(path: Path) -> dict:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if provenance["evidence_level"] != "numerical":
        raise AssertionError("method figures must be labeled numerical evidence")
    manifest_path = Path(provenance["manifest"])
    if _sha256(manifest_path) != provenance["manifest_sha256"]:
        raise AssertionError("method-figure manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_sources = {
        entry["id"]: entry["artifact_sha256"] for entry in manifest["entries"]
    }
    if current_sources != provenance["source_artifact_sha256"]:
        raise AssertionError("method-figure source hashes disagree with manifest")
    for output, expected_hash in provenance["figures"].items():
        output_path = Path(output)
        if not output_path.is_file() or _sha256(output_path) != expected_hash:
            raise AssertionError(f"method figure hash mismatch: {output}")
    if len(provenance["figures"]) != 4:
        raise AssertionError("expected PNG and PDF for two method figures")
    return {"verified": True, "figures": len(provenance["figures"])}


if __name__ == "__main__":
    default = Path("docs/paper/figures/phase30-method-figure-provenance.json")
    print(json.dumps(verify_provenance(default), indent=2))
