"""Independently verify the persisted ADR-0033 Phase 44 result."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from femps.algorithms import canonical_exterior_carrier, vmc_observables

try:
    from scripts import benchmark_phase44_n4_explicit_correlation_d_gate as phase44
    from scripts.phase44_reference_comparators import load_frozen_comparators
except ModuleNotFoundError:
    import benchmark_phase44_n4_explicit_correlation_d_gate as phase44
    from phase44_reference_comparators import load_frozen_comparators


DEFAULT_ARTIFACT = Path(
    "docs/experiments/results/phase44_n4_explicit_correlation_d_gate.json"
)
DEFAULT_CHECKPOINT_MANIFEST = Path(
    "docs/experiments/results/phase44_optimizer_checkpoint_manifest.json"
)
FLOAT_FIELDS = (
    "energy",
    "kinetic_energy",
    "trap_energy",
    "interaction_energy",
    "energy_variance",
    "energy_standard_error",
    "ess_standard_error",
    "blocking_standard_error",
    "chain_mean_standard_error",
    "effective_sample_size",
    "rhat",
)


def _normalized_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _assert_close(actual: float, expected: float, label: str) -> float:
    difference = abs(actual - expected)
    if not math.isclose(actual, expected, rel_tol=3e-13, abs_tol=3e-14):
        raise AssertionError(f"{label} mismatch: {actual!r} != {expected!r}")
    return difference


def _assert_tensor_equal(actual: torch.Tensor, expected: Any, label: str) -> None:
    reference = (
        expected.detach().to(dtype=torch.float64)
        if isinstance(expected, torch.Tensor)
        else torch.tensor(expected, dtype=torch.float64)
    )
    if actual.shape != reference.shape or not torch.equal(actual, reference):
        raise AssertionError(f"{label} tensor mismatch")


def _verify_observables(
    recomputed: dict[str, Any], stored: dict[str, Any], label: str
) -> float:
    maximum = 0.0
    for field in FLOAT_FIELDS:
        maximum = max(
            maximum,
            _assert_close(recomputed[field], stored[field], f"{label}.{field}"),
        )
    for field in ("blocking_size", "blocks_per_chain"):
        if recomputed[field] != stored[field]:
            raise AssertionError(f"{label}.{field} mismatch")
    for field in ("integrated_autocorrelation_times", "chain_means"):
        _assert_tensor_equal(
            torch.tensor(recomputed[field], dtype=torch.float64),
            stored[field],
            f"{label}.{field}",
        )
    for field in ("antisymmetry_residual", "correlator_symmetry_residual"):
        maximum = max(
            maximum,
            _assert_close(
                recomputed["symmetry"][field],
                stored["symmetry"][field],
                f"{label}.symmetry.{field}",
            ),
        )
    return maximum


def _verify_optimizer_checkpoints(
    artifact_path: Path,
    artifact: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported Phase 44 checkpoint manifest")
    if _normalized_sha256(artifact_path) != manifest[
        "phase44_artifact_normalized_sha256"
    ]:
        raise AssertionError("Phase 44 artifact changed after checkpoint manifest")
    records = {
        (record["D"], record["lineage"]): record
        for record in artifact["optimizer_runs"]
    }
    payloads = {}
    for checkpoint in manifest["checkpoints"]:
        path = Path(checkpoint["path"])
        if path.stat().st_size != checkpoint["bytes"]:
            raise AssertionError(f"checkpoint size mismatch: {path}")
        if _raw_sha256(path) != checkpoint["sha256"]:
            raise AssertionError(f"checkpoint hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        key = (checkpoint["D"], checkpoint["lineage"], checkpoint["role"])
        payloads[key] = payload
        if payload["schema_version"] != 1:
            raise AssertionError(f"checkpoint schema mismatch: {path}")
        seed = phase44.OPTIMIZER_SEEDS[checkpoint["D"]][checkpoint["lineage"] - 1]
        if payload["config"] != asdict(phase44._optimizer_config(seed)):
            raise AssertionError(f"checkpoint optimizer config mismatch: {path}")
        if payload["completed_steps"] != 100:
            raise AssertionError(f"incomplete optimizer checkpoint: {path}")
        if checkpoint["role"] != "clean_control":
            record = records[(checkpoint["D"], checkpoint["lineage"])]
            _assert_tensor_equal(
                payload["raw_orbitals"], record["raw_orbitals"],
                f"checkpoint D{checkpoint['D']} lineage{checkpoint['lineage']} raw",
            )
            _assert_tensor_equal(
                payload["amplitudes"], record["amplitudes"],
                f"checkpoint D{checkpoint['D']} lineage{checkpoint['lineage']} amplitudes",
            )
            if payload["history"] != record["history"]:
                raise AssertionError("optimizer checkpoint history mismatch")
            if payload["accepted_proposals"] != record["accepted_proposals"]:
                raise AssertionError("optimizer accepted-proposal mismatch")
            if payload["total_proposals"] != record["total_proposals"]:
                raise AssertionError("optimizer total-proposal mismatch")
    resumed = payloads[(6, 1, "forced_resume")]
    clean = payloads[(6, 1, "clean_control")]
    exact_fields = (
        "completed_steps",
        "history",
        "accepted_proposals",
        "total_proposals",
    )
    if any(resumed[field] != clean[field] for field in exact_fields):
        raise AssertionError("D6 resumed/clean scalar trajectory mismatch")
    tensor_fields = (
        "raw_orbitals",
        "amplitudes",
        "positions",
        "generator_state",
        "raw_first_moment",
        "raw_second_moment",
        "amplitude_first_moment",
        "amplitude_second_moment",
    )
    if any(not torch.equal(resumed[field], clean[field]) for field in tensor_fields):
        raise AssertionError("D6 resumed/clean tensor trajectory mismatch")
    return {
        "verified_checkpoint_count": len(manifest["checkpoints"]),
        "forced_resume_clean_bitwise_equal": True,
    }


def verify(
    artifact_path: Path = DEFAULT_ARTIFACT,
    checkpoint_manifest_path: Path = DEFAULT_CHECKPOINT_MANIFEST,
) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported Phase 44 artifact schema")
    if artifact.get("evidence_level") != "preregistered internal numerical evidence":
        raise ValueError("Phase 44 evidence level changed")
    boundary = artifact.get("scientific_boundary", "")
    for phrase in ("not external replication", "scalability/superiority", "Paper B"):
        if phrase not in boundary:
            raise ValueError(f"Phase 44 boundary is missing {phrase!r}")
    for source, expected_hash in artifact["source_hashes"].items():
        if _normalized_sha256(Path(source)) != expected_hash:
            raise AssertionError(f"Phase 44 source hash mismatch: {source}")

    ledger_path = Path(artifact["reference_firewall"]["selection_ledger"])
    if _normalized_sha256(ledger_path) != artifact["reference_firewall"][
        "selection_ledger_sha256"
    ]:
        raise AssertionError("Phase 44 selection ledger hash mismatch")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    reconstructed_ledger = phase44._selection_ledger(
        Path(artifact["reference_firewall"]["fixture_loaded_before_selection"]),
        artifact["optimizer_runs"],
        artifact["selection_runs"],
    )
    if ledger != reconstructed_ledger:
        raise AssertionError("Phase 44 selection ledger reconstruction mismatch")
    if ledger["choices"] != artifact["selection_choices"]:
        raise AssertionError("Phase 44 selected lineages changed")
    if not artifact["reference_firewall"][
        "comparator_artifacts_opened_only_after_ledger_hash"
    ]:
        raise AssertionError("Phase 44 reference firewall was not asserted")

    checkpoint_result = _verify_optimizer_checkpoints(
        artifact_path, artifact, checkpoint_manifest_path
    )
    optimizer_records = {
        (record["D"], record["lineage"]): record
        for record in artifact["optimizer_runs"]
    }
    for record in optimizer_records.values():
        orbitals = torch.tensor(record["orbitals"], dtype=torch.float64)
        amplitudes = torch.tensor(record["amplitudes"], dtype=torch.float64)
        exponents = torch.tensor(record["exponents"], dtype=torch.float64)
        raw = torch.tensor(record["raw_orbitals"], dtype=torch.float64)
        _assert_tensor_equal(canonical_exterior_carrier(raw), orbitals, "optimizer QR")
        if _tensor_sha256(orbitals, amplitudes, exponents) != record["state_sha256"]:
            raise AssertionError("optimizer state hash mismatch")
        gates = phase44._optimizer_gate(record)
        if gates != record["gates"]:
            raise AssertionError("optimizer gate decision mismatch")

    selection_records = {
        (record["D"], record["lineage"]): record
        for record in artifact["selection_runs"]
    }
    confirmation_records = {
        (record["D"], record["confirmation_index"]): record
        for record in artifact["confirmation_runs"]
    }
    choices = {choice["D"]: choice for choice in artifact["selection_choices"]}
    maximum_observable_difference = 0.0
    for basis_order in phase44.D_AXIS:
        archive_record = artifact["sample_archives"][str(basis_order)]
        archive_path = Path(archive_record["path"])
        if _raw_sha256(archive_path) != archive_record["sha256"]:
            raise AssertionError(f"D{basis_order} sample archive hash mismatch")
        with np.load(archive_path, allow_pickle=False) as archive:
            shapes = {key: list(archive[key].shape) for key in archive.files}
            if shapes != archive_record["arrays"]:
                raise AssertionError(f"D{basis_order} sample archive shapes changed")
            expected_keys = {
                "selection_lineage1",
                "selection_lineage2",
                "confirmation1",
                "confirmation2",
            }
            if set(archive.files) != expected_keys:
                raise AssertionError(f"D{basis_order} sample archive keys changed")
            for lineage in (1, 2):
                record = selection_records[(basis_order, lineage)]
                state = optimizer_records[(basis_order, lineage)]
                samples = torch.from_numpy(
                    archive[f"selection_lineage{lineage}"].copy()
                )
                recomputed = vmc_observables(
                    phase44._selection_config(record["seed"]),
                    torch.tensor(state["orbitals"], dtype=torch.float64),
                    torch.tensor(state["amplitudes"], dtype=torch.float64),
                    torch.tensor(state["exponents"], dtype=torch.float64),
                    samples,
                )
                maximum_observable_difference = max(
                    maximum_observable_difference,
                    _verify_observables(
                        recomputed, record, f"D{basis_order}.selection{lineage}"
                    ),
                )
                merged = dict(record)
                merged.update(recomputed)
                gates = phase44._evaluation_gate(merged)
                if gates != record["gates"]:
                    raise AssertionError("selection evaluation gate mismatch")
                del samples
            selected_lineage = choices[basis_order]["selected_lineage"]
            state = optimizer_records[(basis_order, selected_lineage)]
            for confirmation_index in (1, 2):
                record = confirmation_records[(basis_order, confirmation_index)]
                samples = torch.from_numpy(archive[f"confirmation{confirmation_index}"].copy())
                recomputed = vmc_observables(
                    phase44._confirmation_config(record["seed"]),
                    torch.tensor(state["orbitals"], dtype=torch.float64),
                    torch.tensor(state["amplitudes"], dtype=torch.float64),
                    torch.tensor(state["exponents"], dtype=torch.float64),
                    samples,
                )
                maximum_observable_difference = max(
                    maximum_observable_difference,
                    _verify_observables(
                        recomputed,
                        record,
                        f"D{basis_order}.confirmation{confirmation_index}",
                    ),
                )
                merged = dict(record)
                merged.update(recomputed)
                gates = phase44._evaluation_gate(merged)
                if gates != record["gates"]:
                    raise AssertionError("confirmation evaluation gate mismatch")
                del samples

    comparators = load_frozen_comparators()
    if comparators != artifact["comparators"]:
        raise AssertionError("Phase 44 comparator reconstruction mismatch")
    combined = {}
    for basis_order in phase44.D_AXIS:
        records = [
            confirmation_records[(basis_order, index)] for index in (1, 2)
        ]
        combined[str(basis_order)] = phase44._combine_confirmation(records)
    if combined != artifact["combined_confirmations"]:
        raise AssertionError("Phase 44 confirmation aggregation mismatch")

    monotonic = []
    for lower, upper in zip(phase44.D_AXIS, phase44.D_AXIS[1:]):
        lower_record = combined[str(lower)]
        upper_record = combined[str(upper)]
        allowance = 5.0 * math.sqrt(
            lower_record["inverse_variance_standard_error"] ** 2
            + upper_record["inverse_variance_standard_error"] ** 2
        ) + 2e-4
        monotonic.append(
            {
                "lower_D": lower,
                "upper_D": upper,
                "energy_change": upper_record["inverse_variance_energy"]
                - lower_record["inverse_variance_energy"],
                "allowance": allowance,
                "pass": upper_record["inverse_variance_energy"]
                <= lower_record["inverse_variance_energy"] + allowance,
            }
        )
    if monotonic != artifact["D_monotonicity"]:
        raise AssertionError("Phase 44 D-monotonicity decision mismatch")

    point_gates = []
    reference = comparators["d14_numerical_reference"]
    for basis_order in phase44.D_AXIS:
        record = combined[str(basis_order)]
        error = abs(record["inverse_variance_energy"] - reference)
        conservative = error + 5.0 * record["inverse_variance_standard_error"]
        noci_error = comparators["noci_absolute_reference_errors"][str(basis_order)]
        point_gates.append(
            {
                "D": basis_order,
                "explicit_absolute_reference_error": error,
                "explicit_conservative_error": conservative,
                "noci_absolute_reference_error": noci_error,
                "conservative_error_ratio": conservative / noci_error,
                "pass": conservative <= 0.5 * noci_error,
            }
        )
    if point_gates != artifact["point_advantage_gates"]:
        raise AssertionError("Phase 44 point-advantage decision mismatch")
    consecutive = [
        [first["D"], second["D"]]
        for first, second in zip(point_gates, point_gates[1:])
        if first["pass"] and second["pass"]
    ]
    if consecutive != artifact["consecutive_advantage_pairs"]:
        raise AssertionError("Phase 44 consecutive-advantage pairs changed")
    acceptance = {
        "all_optimizer_gates_pass": all(
            record["gates"]["all_pass"] for record in optimizer_records.values()
        ),
        "forced_resume_clean_pass": checkpoint_result[
            "forced_resume_clean_bitwise_equal"
        ],
        "all_selection_gates_pass": all(
            record["gates"]["all_pass"] for record in selection_records.values()
        ),
        "all_confirmation_gates_pass": all(
            record["gates"]["all_pass"]
            for record in confirmation_records.values()
        ),
        "all_confirmation_seed_agreements_pass": all(
            record["seed_agreement_pass"] for record in combined.values()
        ),
        "D_monotonicity_pass": all(record["pass"] for record in monotonic),
        "two_consecutive_D_advantage_pass": bool(consecutive),
    }
    acceptance["phase44_interacting_d_gate_pass"] = all(acceptance.values())
    if acceptance != artifact["acceptance"]:
        raise AssertionError("Phase 44 aggregate acceptance mismatch")
    failed_selection = [
        {
            "D": record["D"],
            "lineage": record["lineage"],
            "failed_gates": [
                key
                for key, passed in record["gates"].items()
                if key != "all_pass" and not passed
            ],
        }
        for record in selection_records.values()
        if not record["gates"]["all_pass"]
    ]
    return {
        "verified": True,
        "phase44_interacting_d_gate_pass": acceptance[
            "phase44_interacting_d_gate_pass"
        ],
        "two_consecutive_D_advantage_pass": acceptance[
            "two_consecutive_D_advantage_pass"
        ],
        "consecutive_advantage_pairs": consecutive,
        "failed_selection_evaluations": failed_selection,
        "all_confirmation_gates_pass": acceptance[
            "all_confirmation_gates_pass"
        ],
        "maximum_observable_difference": maximum_observable_difference,
        "verified_checkpoint_count": checkpoint_result[
            "verified_checkpoint_count"
        ],
        "external_independent_replication_complete": False,
        "paper_b_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST
    )
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.input, arguments.checkpoint_manifest), indent=2))


if __name__ == "__main__":
    main()
