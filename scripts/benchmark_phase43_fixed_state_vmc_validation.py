"""Run the ADR-0032 fixed-state VMC validation before N=4 production."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from femps.algorithms import (
    CorrelatedExteriorVMCConfig,
    canonical_exterior_carrier,
    canonical_lowest_orbitals,
    canonical_two_orbital_carrier,
    correlated_two_fermion_observables,
    run_correlated_exterior_vmc,
    vmc_energy_gradient,
)
from femps.benchmarks import ProcessRSSMonitor


DEFAULT_SOURCE = Path(
    "docs/experiments/results/phase40_explicit_correlation_gate.json"
)
DEFAULT_OUTPUT = Path(
    "docs/experiments/results/phase43_fixed_state_vmc_validation.json"
)
DEFAULT_SAMPLE_ARCHIVE = Path(
    "docs/experiments/results/phase43_fixed_state_vmc_samples.npz"
)
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/phase43_fixed_state_vmc_validation")
N2_SEEDS = (43011, 43012)
ENERGY_FLOOR = 2e-4
ENERGY_SEED_FLOOR = 1e-4
GRADIENT_FLOOR = 5e-3
SYMMETRY_TOLERANCE = 1e-12


def _normalized_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _phase40_fixture(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        point
        for point in source["correlated_points"]
        if point["D"] == 4 and point["P"] == 3 and point["seed"] == 40001
    ]
    if len(matches) != 1:
        raise ValueError("Phase 40 D4/P3/seed40001 validation fixture is missing")
    return matches[0]


def _deterministic_gradient(
    raw_carrier: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
) -> dict[str, Any]:
    raw = raw_carrier.detach().clone().contiguous().requires_grad_(True)
    amp = amplitudes.detach().clone().contiguous().requires_grad_(True)
    energy = correlated_two_fermion_observables(
        canonical_two_orbital_carrier(raw),
        amp,
        exponents,
        quadrature_order=160,
        coupling=1.0,
        softening=1.0,
    ).energy
    orbital_gradient, amplitude_gradient = torch.autograd.grad(energy, (raw, amp))
    return {
        "energy": float(energy.detach()),
        "orbital_gradient": orbital_gradient.detach(),
        "amplitude_gradient": amplitude_gradient.detach(),
    }


def _n2_config(seed: int) -> CorrelatedExteriorVMCConfig:
    return CorrelatedExteriorVMCConfig(
        particles=2,
        chains=32,
        burn_in_sweeps=500,
        samples_per_chain=3000,
        thinning_sweeps=3,
        proposal_scale=0.8,
        seed=seed,
        max_autocorrelation_lag=100,
        checkpoint_every=500,
        coupling=1.0,
        softening=1.0,
    )


def _n4_config() -> CorrelatedExteriorVMCConfig:
    return CorrelatedExteriorVMCConfig(
        particles=4,
        chains=16,
        burn_in_sweeps=100,
        samples_per_chain=200,
        thinning_sweeps=2,
        proposal_scale=0.7,
        seed=43021,
        max_autocorrelation_lag=50,
        checkpoint_every=40,
        coupling=0.0,
    )


def _without_samples(result: dict[str, Any]) -> dict[str, Any]:
    record = {key: value for key, value in result.items() if key != "samples"}
    if record.get("checkpoint_path") is not None:
        record["checkpoint_path"] = record["checkpoint_path"].replace("\\", "/")
    return record


def _tensor_record(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def _n2_run(
    fixture: dict[str, Any],
    seed: int,
    checkpoint_dir: Path,
    deterministic: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    raw = torch.tensor(fixture["raw_carrier"], dtype=torch.float64)
    orbitals = canonical_exterior_carrier(raw)
    amplitudes = torch.tensor(fixture["amplitudes"], dtype=torch.float64)
    exponents = torch.tensor(fixture["exponents"], dtype=torch.float64)
    config = _n2_config(seed)
    checkpoint = checkpoint_dir / f"n2_d4_p3_seed{seed}.pt"
    started = time.perf_counter()
    with ProcessRSSMonitor() as monitor:
        result = run_correlated_exterior_vmc(
            config,
            orbitals,
            amplitudes,
            exponents,
            checkpoint_path=checkpoint,
        )
        gradient = vmc_energy_gradient(
            config, raw, amplitudes, exponents, result["samples"]
        )
    elapsed = time.perf_counter() - started
    memory = monitor.record()
    orbital_difference = torch.abs(
        gradient["orbital_gradient"] - deterministic["orbital_gradient"]
    )
    amplitude_difference = torch.abs(
        gradient["amplitude_gradient"] - deterministic["amplitude_gradient"]
    )
    orbital_allowance = (
        5.0 * gradient["orbital_chain_standard_error"] + GRADIENT_FLOOR
    )
    amplitude_allowance = (
        5.0 * gradient["amplitude_chain_standard_error"] + GRADIENT_FLOOR
    )
    energy_error = abs(result["energy"] - deterministic["energy"])
    energy_allowance = max(
        5.0 * result["energy_standard_error"], ENERGY_FLOOR
    )
    gates = {
        "energy_pass": energy_error <= energy_allowance,
        "acceptance_pass": 0.15 <= result["acceptance_rate"] <= 0.85,
        "rhat_pass": result["rhat"] <= 1.10,
        "effective_sample_size_pass": result["effective_sample_size"] >= 1000,
        "orbital_gradient_pass": bool(torch.all(orbital_difference <= orbital_allowance)),
        "amplitude_gradient_pass": bool(
            torch.all(amplitude_difference <= amplitude_allowance)
        ),
        "antisymmetry_pass": result["symmetry"]["antisymmetry_residual"]
        <= SYMMETRY_TOLERANCE,
        "correlator_symmetry_pass": result["symmetry"][
            "correlator_symmetry_residual"
        ]
        <= SYMMETRY_TOLERANCE,
    }
    gates["all_pass"] = all(gates.values())
    record = _without_samples(result)
    record.update(
        {
            "D": 4,
            "P": 3,
            "seed": seed,
            "fixture_source_seed": 40001,
            "deterministic_energy": deterministic["energy"],
            "energy_error": energy_error,
            "energy_allowance": energy_allowance,
            "gradient": {
                "vmc_orbital": _tensor_record(gradient["orbital_gradient"]),
                "deterministic_orbital": _tensor_record(
                    deterministic["orbital_gradient"]
                ),
                "orbital_chain_standard_error": _tensor_record(
                    gradient["orbital_chain_standard_error"]
                ),
                "orbital_absolute_difference": _tensor_record(orbital_difference),
                "orbital_allowance": _tensor_record(orbital_allowance),
                "vmc_amplitude": _tensor_record(gradient["amplitude_gradient"]),
                "deterministic_amplitude": _tensor_record(
                    deterministic["amplitude_gradient"]
                ),
                "amplitude_chain_standard_error": _tensor_record(
                    gradient["amplitude_chain_standard_error"]
                ),
                "amplitude_absolute_difference": _tensor_record(
                    amplitude_difference
                ),
                "amplitude_allowance": _tensor_record(amplitude_allowance),
            },
            "gates": gates,
            "elapsed_seconds": elapsed,
            "cpu_memory": memory.as_dict(),
            "peak_cpu_rss_bytes": memory.peak_rss_bytes,
            "materialization": {
                "coordinate_samples_stored": int(result["samples"].numel()),
                "D_to_the_N_tensor": False,
                "full_alternating_coefficient_tensor": False,
                "virtual_paths": 0,
            },
        }
    )
    return record, result["samples"]


def _n4_resume_run(
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], torch.Tensor]:
    config = _n4_config()
    orbitals = canonical_lowest_orbitals(4, 4)
    empty = torch.empty(0, dtype=torch.float64)
    checkpoint = checkpoint_dir / "n4_noninteracting_seed43021.pt"
    partial = run_correlated_exterior_vmc(
        config,
        orbitals,
        empty,
        empty,
        checkpoint_path=checkpoint,
        max_samples_this_call=80,
    )
    resumed = run_correlated_exterior_vmc(
        config,
        orbitals,
        empty,
        empty,
        checkpoint_path=checkpoint,
        resume=True,
    )
    clean = run_correlated_exterior_vmc(config, orbitals, empty, empty)
    samples_identical = torch.equal(resumed["samples"], clean["samples"])
    observables_identical = all(
        resumed[key] == clean[key]
        for key in (
            "energy",
            "energy_variance",
            "energy_standard_error",
            "acceptance_rate",
            "accepted_proposals",
            "total_proposals",
            "effective_sample_size",
            "integrated_autocorrelation_times",
            "rhat",
            "chain_means",
            "symmetry",
        )
    )
    gates = {
        "partial_stopped_at_80": (
            not partial["completed"]
            and partial["completed_samples_per_chain"] == 80
        ),
        "energy_pass": abs(resumed["energy"] - 8.0) <= 1e-12,
        "variance_pass": resumed["energy_variance"] <= 1e-20,
        "resume_samples_identical": samples_identical,
        "resume_observables_identical": observables_identical,
        "antisymmetry_pass": resumed["symmetry"]["antisymmetry_residual"]
        <= SYMMETRY_TOLERANCE,
        "correlator_symmetry_pass": resumed["symmetry"][
            "correlator_symmetry_residual"
        ]
        <= SYMMETRY_TOLERANCE,
    }
    gates["all_pass"] = all(gates.values())
    record = _without_samples(resumed)
    record.update(
        {
            "D": 4,
            "P": 0,
            "seed": 43021,
            "exact_energy": 8.0,
            "energy_error": abs(resumed["energy"] - 8.0),
            "forced_partial_samples_per_chain": partial[
                "completed_samples_per_chain"
            ],
            "gates": gates,
            "materialization": {
                "coordinate_samples_stored": int(resumed["samples"].numel()),
                "D_to_the_N_tensor": False,
                "full_alternating_coefficient_tensor": False,
                "virtual_paths": 0,
            },
        }
    )
    return record, resumed["samples"]


def run(
    source_path: Path,
    output_path: Path,
    sample_archive_path: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    if output_path.exists() or sample_archive_path.exists():
        raise FileExistsError("Phase 43 validation output already exists")
    fixture = _phase40_fixture(source_path)
    raw = torch.tensor(fixture["raw_carrier"], dtype=torch.float64)
    amplitudes = torch.tensor(fixture["amplitudes"], dtype=torch.float64)
    exponents = torch.tensor(fixture["exponents"], dtype=torch.float64)
    deterministic = _deterministic_gradient(raw, amplitudes, exponents)
    n2_records = []
    sample_arrays: dict[str, np.ndarray] = {}
    for seed in N2_SEEDS:
        record, samples = _n2_run(
            fixture, seed, checkpoint_dir, deterministic
        )
        n2_records.append(record)
        sample_arrays[f"n2_seed{seed}"] = samples.detach().cpu().numpy()
        print(
            f"N2 seed={seed} energy={record['energy']:.12f} "
            f"se={record['energy_standard_error']:.3e} "
            f"ESS={record['effective_sample_size']:.1f}",
            flush=True,
        )
    n4_record, n4_samples = _n4_resume_run(checkpoint_dir)
    sample_arrays["n4_seed43021"] = n4_samples.detach().cpu().numpy()
    sample_archive_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_archive_path, **sample_arrays)

    energy_seed_difference = abs(n2_records[0]["energy"] - n2_records[1]["energy"])
    energy_seed_allowance = (
        5.0
        * math.sqrt(
            n2_records[0]["energy_standard_error"] ** 2
            + n2_records[1]["energy_standard_error"] ** 2
        )
        + ENERGY_SEED_FLOOR
    )
    acceptance = {
        "all_n2_point_gates_pass": all(
            record["gates"]["all_pass"] for record in n2_records
        ),
        "n2_distinct_seed_agreement_pass": energy_seed_difference
        <= energy_seed_allowance,
        "n4_noninteracting_resume_pass": n4_record["gates"]["all_pass"],
    }
    acceptance["phase43_fixed_state_validation_pass"] = all(acceptance.values())
    artifact = {
        "schema_version": 1,
        "experiment": "phase43_fixed_state_correlated_exterior_vmc_validation",
        "evidence_level": "preregistered estimator validation",
        "scientific_boundary": "fixed-state N2 deterministic-truth and N4 noninteracting checks; not an interacting N4 result, scalable solver claim, external replication, or Paper B",
        "adr": "docs/decisions/0032-preregister-phase43-fixed-state-vmc-validation.md",
        "phase40_fixture": {
            "artifact": source_path.as_posix(),
            "D": 4,
            "P": 3,
            "seed": 40001,
            "selection_disclosure": "post-Phase40 stable implementation-validation fixture, not an independently predicted physics point",
            "raw_carrier": fixture["raw_carrier"],
            "amplitudes": fixture["amplitudes"],
            "exponents": fixture["exponents"],
        },
        "frozen_thresholds": {
            "energy_sigma_multiplier": 5.0,
            "energy_absolute_floor": ENERGY_FLOOR,
            "distinct_seed_absolute_floor": ENERGY_SEED_FLOOR,
            "gradient_sigma_multiplier": 5.0,
            "gradient_absolute_floor": GRADIENT_FLOOR,
            "acceptance_interval": [0.15, 0.85],
            "maximum_rhat": 1.10,
            "minimum_effective_sample_size": 1000,
            "symmetry_residual": SYMMETRY_TOLERANCE,
            "n4_energy_error": 1e-12,
            "n4_variance": 1e-20,
        },
        "frozen_configs": {
            "n2": {str(seed): asdict(_n2_config(seed)) for seed in N2_SEEDS},
            "n4_noninteracting": asdict(_n4_config()),
        },
        "deterministic_n2_truth": {
            "quadrature_order": 160,
            "energy": deterministic["energy"],
            "orbital_gradient": _tensor_record(
                deterministic["orbital_gradient"]
            ),
            "amplitude_gradient": _tensor_record(
                deterministic["amplitude_gradient"]
            ),
        },
        "n2_runs": n2_records,
        "n2_distinct_seed_comparison": {
            "absolute_energy_difference": energy_seed_difference,
            "allowance": energy_seed_allowance,
            "pass": energy_seed_difference <= energy_seed_allowance,
        },
        "n4_noninteracting": n4_record,
        "sample_archive": {
            "path": sample_archive_path.as_posix(),
            "sha256": _raw_sha256(sample_archive_path),
            "arrays": {
                key: list(value.shape) for key, value in sample_arrays.items()
            },
        },
        "source_hashes": {
            path.as_posix(): _normalized_sha256(path)
            for path in (
                Path("src/femps/algorithms/correlated_exterior_vmc.py"),
                Path("scripts/benchmark_phase43_fixed_state_vmc_validation.py"),
                Path(
                    "docs/decisions/"
                    "0032-preregister-phase43-fixed-state-vmc-validation.md"
                ),
            )
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "dtype": "float64",
        },
        "acceptance": acceptance,
    }
    _write(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-archive", type=Path, default=DEFAULT_SAMPLE_ARCHIVE)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    arguments = parser.parse_args()
    result = run(
        arguments.source,
        arguments.output,
        arguments.sample_archive,
        arguments.checkpoint_dir,
    )
    print(json.dumps(result["acceptance"], indent=2))


if __name__ == "__main__":
    main()
