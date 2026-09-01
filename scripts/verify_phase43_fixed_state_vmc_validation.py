"""Verify the persisted ADR-0032 fixed-state VMC validation artifact."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch

from femps.algorithms import (
    canonical_exterior_carrier,
    canonical_lowest_orbitals,
    run_correlated_exterior_vmc,
    vmc_energy_gradient,
    vmc_observables,
)

try:
    from scripts.benchmark_phase43_fixed_state_vmc_validation import (
        DEFAULT_OUTPUT,
        ENERGY_FLOOR,
        ENERGY_SEED_FLOOR,
        GRADIENT_FLOOR,
        N2_SEEDS,
        SYMMETRY_TOLERANCE,
        _deterministic_gradient,
        _n2_config,
        _n4_config,
    )
except ModuleNotFoundError:
    from benchmark_phase43_fixed_state_vmc_validation import (
        DEFAULT_OUTPUT,
        ENERGY_FLOOR,
        ENERGY_SEED_FLOOR,
        GRADIENT_FLOOR,
        N2_SEEDS,
        SYMMETRY_TOLERANCE,
        _deterministic_gradient,
        _n2_config,
        _n4_config,
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


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-14):
        raise AssertionError(f"{label} mismatch: {actual!r} != {expected!r}")


def _assert_tensor_close(actual: torch.Tensor, expected: Any, label: str) -> None:
    reference = (
        expected.detach().to(dtype=torch.float64)
        if isinstance(expected, torch.Tensor)
        else torch.tensor(expected, dtype=torch.float64)
    )
    if actual.shape != reference.shape or not torch.allclose(
        actual, reference, rtol=2e-13, atol=2e-14
    ):
        maximum = (
            float(torch.max(torch.abs(actual - reference)))
            if actual.shape == reference.shape and actual.numel()
            else math.inf
        )
        raise AssertionError(f"{label} mismatch (maximum difference {maximum:.3e})")


def _verify_observables(
    recomputed: dict[str, Any], stored: dict[str, Any], label: str
) -> None:
    for field in FLOAT_FIELDS:
        _assert_close(recomputed[field], stored[field], f"{label}.{field}")
    if recomputed["blocking_size"] != stored["blocking_size"]:
        raise AssertionError(f"{label}.blocking_size mismatch")
    if recomputed["blocks_per_chain"] != stored["blocks_per_chain"]:
        raise AssertionError(f"{label}.blocks_per_chain mismatch")
    for field in ("integrated_autocorrelation_times", "chain_means"):
        _assert_tensor_close(
            torch.tensor(recomputed[field], dtype=torch.float64),
            stored[field],
            f"{label}.{field}",
        )
    for field in ("antisymmetry_residual", "correlator_symmetry_residual"):
        _assert_close(
            recomputed["symmetry"][field],
            stored["symmetry"][field],
            f"{label}.symmetry.{field}",
        )


def _n2_gates(
    record: dict[str, Any],
    deterministic: dict[str, Any],
    gradient: dict[str, torch.Tensor],
) -> dict[str, bool]:
    orbital_difference = torch.abs(
        gradient["orbital_gradient"] - deterministic["orbital_gradient"]
    )
    amplitude_difference = torch.abs(
        gradient["amplitude_gradient"] - deterministic["amplitude_gradient"]
    )
    orbital_allowance = 5.0 * gradient["orbital_chain_standard_error"] + GRADIENT_FLOOR
    amplitude_allowance = (
        5.0 * gradient["amplitude_chain_standard_error"] + GRADIENT_FLOOR
    )
    energy_allowance = max(5.0 * record["energy_standard_error"], ENERGY_FLOOR)
    gates = {
        "energy_pass": abs(record["energy"] - deterministic["energy"])
        <= energy_allowance,
        "acceptance_pass": 0.15 <= record["acceptance_rate"] <= 0.85,
        "rhat_pass": record["rhat"] <= 1.10,
        "effective_sample_size_pass": record["effective_sample_size"] >= 1000,
        "orbital_gradient_pass": bool(
            torch.all(orbital_difference <= orbital_allowance)
        ),
        "amplitude_gradient_pass": bool(
            torch.all(amplitude_difference <= amplitude_allowance)
        ),
        "antisymmetry_pass": record["symmetry"]["antisymmetry_residual"]
        <= SYMMETRY_TOLERANCE,
        "correlator_symmetry_pass": record["symmetry"][
            "correlator_symmetry_residual"
        ]
        <= SYMMETRY_TOLERANCE,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def verify(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported Phase 43 artifact schema")
    if artifact.get("evidence_level") != "preregistered estimator validation":
        raise ValueError("Phase 43 evidence boundary changed")
    boundary = artifact.get("scientific_boundary", "")
    for phrase in ("not an interacting N4 result", "scalable solver claim", "Paper B"):
        if phrase not in boundary:
            raise ValueError(f"Phase 43 scientific boundary is missing {phrase!r}")

    for source, expected_hash in artifact["source_hashes"].items():
        if _normalized_sha256(Path(source)) != expected_hash:
            raise AssertionError(f"source hash mismatch: {source}")
    archive_path = Path(artifact["sample_archive"]["path"])
    if _raw_sha256(archive_path) != artifact["sample_archive"]["sha256"]:
        raise AssertionError("Phase 43 sample archive hash mismatch")

    expected_configs = {
        "n2": {str(seed): asdict(_n2_config(seed)) for seed in N2_SEEDS},
        "n4_noninteracting": asdict(_n4_config()),
    }
    if artifact.get("frozen_configs") != expected_configs:
        raise AssertionError("Phase 43 frozen sampler config mismatch")

    fixture = artifact["phase40_fixture"]
    if (fixture["D"], fixture["P"], fixture["seed"]) != (4, 3, 40001):
        raise AssertionError("Phase 43 fixture identity changed")
    raw = torch.tensor(fixture["raw_carrier"], dtype=torch.float64)
    amplitudes = torch.tensor(fixture["amplitudes"], dtype=torch.float64)
    exponents = torch.tensor(fixture["exponents"], dtype=torch.float64)
    orbitals = canonical_exterior_carrier(raw)
    deterministic = _deterministic_gradient(raw, amplitudes, exponents)
    truth = artifact["deterministic_n2_truth"]
    _assert_close(deterministic["energy"], truth["energy"], "deterministic.energy")
    _assert_tensor_close(
        deterministic["orbital_gradient"], truth["orbital_gradient"],
        "deterministic.orbital_gradient",
    )
    _assert_tensor_close(
        deterministic["amplitude_gradient"], truth["amplitude_gradient"],
        "deterministic.amplitude_gradient",
    )

    n2_passes = []
    maximum_observable_difference = 0.0
    maximum_gradient_difference = 0.0
    with np.load(archive_path, allow_pickle=False) as sample_archive:
        if set(sample_archive.files) != {
            *(f"n2_seed{seed}" for seed in N2_SEEDS),
            "n4_seed43021",
        }:
            raise AssertionError("Phase 43 sample archive arrays changed")
        actual_shapes = {
            key: list(sample_archive[key].shape) for key in sample_archive.files
        }
        if actual_shapes != artifact["sample_archive"]["arrays"]:
            raise AssertionError("Phase 43 sample archive shape mismatch")
        records_by_seed = {record["seed"]: record for record in artifact["n2_runs"]}
        if set(records_by_seed) != set(N2_SEEDS):
            raise AssertionError("Phase 43 N2 seed set changed")
        for seed in N2_SEEDS:
            record = records_by_seed[seed]
            config = _n2_config(seed)
            samples = torch.from_numpy(sample_archive[f"n2_seed{seed}"].copy())
            recomputed = vmc_observables(
                config, orbitals, amplitudes, exponents, samples
            )
            _verify_observables(recomputed, record, f"n2_seed{seed}")
            for field in FLOAT_FIELDS:
                maximum_observable_difference = max(
                    maximum_observable_difference,
                    abs(recomputed[field] - record[field]),
                )
            gradient = vmc_energy_gradient(
                config, raw, amplitudes, exponents, samples
            )
            stored_gradient = record["gradient"]
            for key, stored_key in (
                ("orbital_gradient", "vmc_orbital"),
                ("amplitude_gradient", "vmc_amplitude"),
                ("orbital_chain_standard_error", "orbital_chain_standard_error"),
                ("amplitude_chain_standard_error", "amplitude_chain_standard_error"),
            ):
                reference = torch.tensor(stored_gradient[stored_key], dtype=torch.float64)
                maximum_gradient_difference = max(
                    maximum_gradient_difference,
                    float(torch.max(torch.abs(gradient[key] - reference)))
                    if reference.numel()
                    else 0.0,
                )
                _assert_tensor_close(gradient[key], reference, f"n2_seed{seed}.{key}")
            gates = _n2_gates(record, deterministic, gradient)
            if gates != record["gates"]:
                raise AssertionError(f"N2 seed {seed} gate decision mismatch")
            n2_passes.append(gates["all_pass"])

        n4_config = _n4_config()
        n4_orbitals = canonical_lowest_orbitals(4, 4)
        empty = torch.empty(0, dtype=torch.float64)
        n4_samples = torch.from_numpy(sample_archive["n4_seed43021"].copy())
        n4_recomputed = vmc_observables(
            n4_config, n4_orbitals, empty, empty, n4_samples
        )
        n4_record = artifact["n4_noninteracting"]
        _verify_observables(n4_recomputed, n4_record, "n4_noninteracting")

        with TemporaryDirectory(prefix="femps_phase43_verify_") as temporary:
            checkpoint = Path(temporary) / "n4.pt"
            partial = run_correlated_exterior_vmc(
                n4_config,
                n4_orbitals,
                empty,
                empty,
                checkpoint_path=checkpoint,
                max_samples_this_call=80,
            )
            resumed = run_correlated_exterior_vmc(
                n4_config,
                n4_orbitals,
                empty,
                empty,
                checkpoint_path=checkpoint,
                resume=True,
            )
            clean = run_correlated_exterior_vmc(
                n4_config, n4_orbitals, empty, empty
            )
        n4_resume_pass = (
            not partial["completed"]
            and partial["completed_samples_per_chain"] == 80
            and torch.equal(resumed["samples"], clean["samples"])
            and torch.equal(resumed["samples"], n4_samples)
            and abs(resumed["energy"] - 8.0) <= 1e-12
            and resumed["energy_variance"] <= 1e-20
            and resumed["symmetry"]["antisymmetry_residual"] <= SYMMETRY_TOLERANCE
            and resumed["symmetry"]["correlator_symmetry_residual"]
            <= SYMMETRY_TOLERANCE
        )

    records = artifact["n2_runs"]
    seed_difference = abs(records[0]["energy"] - records[1]["energy"])
    seed_allowance = 5.0 * math.sqrt(
        records[0]["energy_standard_error"] ** 2
        + records[1]["energy_standard_error"] ** 2
    ) + ENERGY_SEED_FLOOR
    seed_pass = seed_difference <= seed_allowance
    acceptance = {
        "all_n2_point_gates_pass": all(n2_passes),
        "n2_distinct_seed_agreement_pass": seed_pass,
        "n4_noninteracting_resume_pass": n4_resume_pass,
    }
    acceptance["phase43_fixed_state_validation_pass"] = all(acceptance.values())
    if acceptance != artifact["acceptance"]:
        raise AssertionError("Phase 43 aggregate acceptance mismatch")
    return {
        "verified": True,
        "phase43_fixed_state_validation_pass": acceptance[
            "phase43_fixed_state_validation_pass"
        ],
        "maximum_observable_difference": maximum_observable_difference,
        "maximum_gradient_difference": maximum_gradient_difference,
        "external_independent_replication_complete": False,
        "interacting_n4_complete": False,
        "paper_b_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.input), indent=2))


if __name__ == "__main__":
    main()
