"""Verify the reference-firewalled Phase 44 K=1 initialization fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_FIXTURE = Path(
    "docs/experiments/results/phase44_phase37_k1_initialization.json"
)


def _normalized_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def verify(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != 1:
        raise ValueError("unsupported Phase 44 initialization fixture schema")
    if fixture.get("source_terms") != 1 or fixture.get("source_seed") != 3701:
        raise AssertionError("Phase 44 initialization is not the frozen K1 source")
    forbidden = {"energy", "reference_energy", "noci_energy", "ci_energy", "k4"}
    present = {key.lower() for key in fixture}
    if forbidden & present:
        raise AssertionError("reference/comparator data leaked into initialization fixture")
    source_path = Path(fixture["source_artifact"])
    if _normalized_sha256(source_path) != fixture["source_normalized_sha256"]:
        raise AssertionError("Phase 37 source artifact hash mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    matching = [
        record for record in source["stage_orbitals"] if record["terms"] == 1
    ]
    if len(matching) != 1 or len(matching[0]["values"]) != 1:
        raise AssertionError("Phase 37 K1 carrier is ambiguous")
    encoded = torch.tensor(matching[0]["values"][0], dtype=torch.float64)
    if encoded.shape != (6, 4, 2):
        raise AssertionError("Phase 37 K1 carrier shape changed")
    if not torch.equal(encoded[..., 1], torch.zeros((6, 4), dtype=torch.float64)):
        raise AssertionError("Phase 37 K1 carrier is no longer exactly real")
    carrier = torch.tensor(fixture["carrier"], dtype=torch.float64)
    if carrier.shape != (6, 4) or not torch.equal(carrier, encoded[..., 0]):
        raise AssertionError("Phase 44 fixture does not exactly reproduce Phase 37 K1")
    gram_residual = float(
        torch.max(
            torch.abs(
                carrier.mT @ carrier - torch.eye(4, dtype=torch.float64)
            )
        )
    )
    if gram_residual > 2e-15:
        raise AssertionError("Phase 44 initialization carrier lost orthonormality")
    return {
        "verified": True,
        "source_terms": 1,
        "source_seed": 3701,
        "carrier_shape": list(carrier.shape),
        "maximum_gram_residual": gram_residual,
        "reference_fields_present": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_FIXTURE)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.input), indent=2))


if __name__ == "__main__":
    main()
