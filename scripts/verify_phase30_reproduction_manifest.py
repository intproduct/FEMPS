"""Independently verify hashes and registered artifact verifiers in Phase 30."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

from femps.algorithms import (
    ADAPTIVE_DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION,
    ADAPTIVE_DIAGONAL_PATH_RESULT_SCHEMA_VERSION,
    DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION,
    DIAGONAL_PATH_RESULT_SCHEMA_VERSION,
    SLATER_SOURCE_CHECKPOINT_SCHEMA_VERSION,
    SLATER_SOURCE_RESULT_SCHEMA_VERSION,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise AssertionError("unsupported reproduction manifest schema")
    contract = manifest["solver_contract"]
    if contract["result_schema_version"] != DIAGONAL_PATH_RESULT_SCHEMA_VERSION:
        raise AssertionError("result schema disagrees with the public solver")
    if (
        contract["checkpoint_schema_version"]
        != DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION
    ):
        raise AssertionError("checkpoint schema disagrees with the public solver")
    if (
        contract["adaptive_result_schema_version"]
        != ADAPTIVE_DIAGONAL_PATH_RESULT_SCHEMA_VERSION
    ):
        raise AssertionError("adaptive result schema disagrees with the public solver")
    if (
        contract["adaptive_checkpoint_schema_version"]
        != ADAPTIVE_DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION
    ):
        raise AssertionError(
            "adaptive checkpoint schema disagrees with the public solver"
        )
    if (
        contract["slater_source_result_schema_version"]
        != SLATER_SOURCE_RESULT_SCHEMA_VERSION
    ):
        raise AssertionError("Slater-source result schema disagrees with the solver")
    if (
        contract["slater_source_checkpoint_schema_version"]
        != SLATER_SOURCE_CHECKPOINT_SCHEMA_VERSION
    ):
        raise AssertionError(
            "Slater-source checkpoint schema disagrees with the solver"
        )
    if not Path(contract["document"]).is_file():
        raise AssertionError("solver contract document is missing")

    ids = []
    results = []
    for entry in manifest["entries"]:
        ids.append(entry["id"])
        artifact_path = Path(entry["artifact"])
        if not artifact_path.is_file():
            raise AssertionError(f"missing artifact: {artifact_path}")
        if _sha256(artifact_path) != entry["artifact_sha256"]:
            raise AssertionError(f"artifact hash mismatch: {entry['id']}")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if payload["schema_version"] != entry["artifact_schema_version"]:
            raise AssertionError(f"artifact schema mismatch: {entry['id']}")
        if payload["evidence_level"] != entry["evidence_level"]:
            raise AssertionError(f"evidence label mismatch: {entry['id']}")
        if entry["evidence_level"] != "numerical":
            raise AssertionError("method manifest artifacts must remain numerical")
        for command_key in ("benchmark_command", "verify_command"):
            script = Path(entry[command_key].split()[-1])
            if not script.is_file():
                raise AssertionError(f"missing command target: {script}")
        module = importlib.import_module(entry["verifier_module"])
        argument = payload if entry["verifier_argument"] == "json_payload" else artifact_path
        result = module.verify_artifact(argument)
        if not result.get("verified"):
            raise AssertionError(f"registered verifier failed: {entry['id']}")
        results.append({"id": entry["id"], "verified": True})
    if len(ids) != len(set(ids)):
        raise AssertionError("manifest claim identifiers must be unique")
    if len(ids) != 14:
        raise AssertionError(
            "restricted-method manifest must contain the fourteen admitted artifacts"
        )
    return {"verified": True, "entries": len(ids), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("docs/experiments/results/phase30_reproduction_manifest.json"),
    )
    args = parser.parse_args()
    print(json.dumps(verify_manifest(args.manifest), indent=2))


if __name__ == "__main__":
    main()
