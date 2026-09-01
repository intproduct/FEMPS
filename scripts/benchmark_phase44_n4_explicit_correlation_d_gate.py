"""Execute the ADR-0033 interacting N=4 explicit-correlation D gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch

from femps.algorithms import (
    CorrelatedExteriorVMCConfig,
    CorrelatedExteriorVMCOptimizerConfig,
    canonical_exterior_carrier,
    run_correlated_exterior_vmc,
    run_correlated_exterior_vmc_optimization,
)
from femps.benchmarks import ProcessRSSMonitor


DEFAULT_FIXTURE = Path(
    "docs/experiments/results/phase44_phase37_k1_initialization.json"
)
DEFAULT_LEDGER = Path(
    "docs/experiments/results/phase44_pre_reference_selection.json"
)
DEFAULT_OUTPUT = Path(
    "docs/experiments/results/phase44_n4_explicit_correlation_d_gate.json"
)
DEFAULT_ARCHIVE_DIR = Path("docs/experiments/results")
DEFAULT_CHECKPOINT_DIR = Path(
    "checkpoints/phase44_n4_explicit_correlation_d_gate"
)
D_AXIS = (4, 6, 8)
OPTIMIZER_SEEDS = {
    4: (44041, 44042),
    6: (44061, 44062),
    8: (44081, 44082),
}
SELECTION_SEEDS = {
    4: (45041, 45042),
    6: (45061, 45062),
    8: (45081, 45082),
}
CONFIRMATION_SEEDS = {
    4: (44241, 44242),
    6: (44261, 44262),
    8: (44281, 44282),
}
EXPONENTS = (0.0625, 0.25, 1.0, 4.0, 16.0)


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


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_reference_free_fixture(path: Path) -> tuple[dict[str, Any], torch.Tensor]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("fixture") != "phase44_phase37_k1_single_slater_initialization":
        raise ValueError("unexpected Phase 44 initialization fixture")
    if fixture.get("source_terms") != 1 or fixture.get("source_seed") != 3701:
        raise ValueError("Phase 44 initialization identity changed")
    forbidden = {"energy", "reference_energy", "noci_energy", "ci_energy", "k4"}
    if forbidden & {key.lower() for key in fixture}:
        raise ValueError("reference data leaked into the Phase 44 fixture")
    carrier = torch.tensor(fixture["carrier"], dtype=torch.float64)
    if carrier.shape != (6, 4):
        raise ValueError("Phase 44 initialization carrier shape changed")
    return fixture, carrier


def _initial_carrier(source: torch.Tensor, basis_order: int) -> torch.Tensor:
    if basis_order == 4:
        raw = source[:4].clone()
    elif basis_order == 6:
        raw = source.clone()
    elif basis_order == 8:
        raw = torch.cat((source, torch.zeros((2, 4), dtype=torch.float64)))
    else:
        raise ValueError("basis order is outside the frozen D axis")
    return canonical_exterior_carrier(raw)


def _optimizer_config(seed: int) -> CorrelatedExteriorVMCOptimizerConfig:
    return CorrelatedExteriorVMCOptimizerConfig(
        particles=4,
        chains=32,
        steps=100,
        burn_in_sweeps=1000,
        rethermalization_sweeps=20,
        samples_per_chain=128,
        thinning_sweeps=2,
        proposal_scale=0.65,
        seed=seed,
        learning_rate=0.01,
        final_learning_rate=0.001,
        gradient_clip_norm=2.0,
        amplitude_bound=1.0,
        checkpoint_every=10,
        max_autocorrelation_lag=100,
        coupling=1.0,
        softening=1.0,
    )


def _selection_config(seed: int) -> CorrelatedExteriorVMCConfig:
    return CorrelatedExteriorVMCConfig(
        particles=4,
        chains=32,
        burn_in_sweeps=1000,
        samples_per_chain=2000,
        thinning_sweeps=3,
        proposal_scale=0.65,
        seed=seed,
        max_autocorrelation_lag=200,
        checkpoint_every=500,
        coupling=1.0,
        softening=1.0,
    )


def _confirmation_config(seed: int) -> CorrelatedExteriorVMCConfig:
    return CorrelatedExteriorVMCConfig(
        particles=4,
        chains=64,
        burn_in_sweeps=2000,
        samples_per_chain=5000,
        thinning_sweeps=4,
        proposal_scale=0.65,
        seed=seed,
        max_autocorrelation_lag=500,
        checkpoint_every=500,
        coupling=1.0,
        softening=1.0,
    )


def _tensor_record(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def _optimizer_record(
    result: dict[str, Any],
    *,
    basis_order: int,
    lineage: int,
    seed: int,
    elapsed_seconds: float,
    memory: dict[str, Any],
) -> dict[str, Any]:
    return {
        "D": basis_order,
        "lineage": lineage,
        "seed": seed,
        "completed": result["completed"],
        "completed_steps": result["completed_steps"],
        "raw_orbitals": _tensor_record(result["raw_orbitals"]),
        "orbitals": _tensor_record(result["orbitals"]),
        "amplitudes": _tensor_record(result["amplitudes"]),
        "exponents": _tensor_record(result["exponents"]),
        "state_sha256": _tensor_sha256(
            result["orbitals"], result["amplitudes"], result["exponents"]
        ),
        "history": result["history"],
        "accepted_proposals": result["accepted_proposals"],
        "total_proposals": result["total_proposals"],
        "acceptance_rate": result["acceptance_rate"],
        "checkpoint_path": result["checkpoint_path"],
        "elapsed_seconds": elapsed_seconds,
        "cpu_memory": memory,
        "peak_cpu_rss_bytes": memory["peak_rss_bytes"],
        "materialization": result["materialization"],
    }


def _run_optimizer(
    basis_order: int,
    lineage: int,
    seed: int,
    raw: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    checkpoint_dir: Path,
    *,
    force_partial: bool = False,
    clean_suffix: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    suffix = "" if clean_suffix is None else f"_{clean_suffix}"
    checkpoint = checkpoint_dir / f"d{basis_order}_lineage{lineage}{suffix}.pt"
    config = _optimizer_config(seed)
    started = time.perf_counter()
    with ProcessRSSMonitor() as monitor:
        if checkpoint.exists():
            result = run_correlated_exterior_vmc_optimization(
                config,
                raw,
                amplitudes,
                exponents,
                checkpoint_path=checkpoint,
                resume=True,
            )
        elif force_partial:
            partial = run_correlated_exterior_vmc_optimization(
                config,
                raw,
                amplitudes,
                exponents,
                checkpoint_path=checkpoint,
                max_steps_this_call=40,
            )
            if partial["completed"] or partial["completed_steps"] != 40:
                raise AssertionError("forced optimizer interruption missed step 40")
            result = run_correlated_exterior_vmc_optimization(
                config,
                raw,
                amplitudes,
                exponents,
                checkpoint_path=checkpoint,
                resume=True,
            )
        else:
            result = run_correlated_exterior_vmc_optimization(
                config,
                raw,
                amplitudes,
                exponents,
                checkpoint_path=checkpoint,
            )
    elapsed = time.perf_counter() - started
    memory = monitor.record().as_dict()
    return result, _optimizer_record(
        result,
        basis_order=basis_order,
        lineage=lineage,
        seed=seed,
        elapsed_seconds=elapsed,
        memory=memory,
    )


def _run_evaluation(
    config: CorrelatedExteriorVMCConfig,
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    with ProcessRSSMonitor() as monitor:
        result = run_correlated_exterior_vmc(
            config,
            orbitals,
            amplitudes,
            exponents,
            checkpoint_path=checkpoint,
            resume=checkpoint.exists(),
        )
    elapsed = time.perf_counter() - started
    memory = monitor.record().as_dict()
    record = {key: value for key, value in result.items() if key != "samples"}
    if record["checkpoint_path"] is not None:
        record["checkpoint_path"] = record["checkpoint_path"].replace("\\", "/")
    record.update(
        {
            "elapsed_seconds": elapsed,
            "cpu_memory": memory,
            "peak_cpu_rss_bytes": memory["peak_rss_bytes"],
            "materialization": {
                "coordinate_samples_stored": int(result["samples"].numel()),
                "D_to_the_N_tensor": False,
                "full_alternating_coefficient_tensor": False,
                "virtual_paths": 0,
            },
        }
    )
    return result, record


def _resume_clean_equal(
    resumed: dict[str, Any], clean: dict[str, Any]
) -> bool:
    return (
        resumed["history"] == clean["history"]
        and resumed["accepted_proposals"] == clean["accepted_proposals"]
        and resumed["total_proposals"] == clean["total_proposals"]
        and torch.equal(resumed["raw_orbitals"], clean["raw_orbitals"])
        and torch.equal(resumed["orbitals"], clean["orbitals"])
        and torch.equal(resumed["amplitudes"], clean["amplitudes"])
    )


def _selection_ledger(
    fixture_path: Path,
    optimizer_records: list[dict[str, Any]],
    selection_records: list[dict[str, Any]],
) -> dict[str, Any]:
    choices = []
    for basis_order in D_AXIS:
        candidates = [
            record for record in selection_records if record["D"] == basis_order
        ]
        if len(candidates) != 2:
            raise AssertionError("selection evaluation axis is incomplete")
        selected = min(candidates, key=lambda record: record["energy"])
        choices.append(
            {
                "D": basis_order,
                "selected_lineage": selected["lineage"],
                "optimizer_seed": selected["optimizer_seed"],
                "selection_seed": selected["seed"],
                "selection_energy": selected["energy"],
                "selection_standard_error": selected["energy_standard_error"],
                "state_sha256": selected["state_sha256"],
            }
        )
    return {
        "schema_version": 1,
        "experiment": "phase44_pre_reference_lineage_selection",
        "reference_firewall": "written before opening D14, CI, or NOCI comparator artifacts",
        "fixture_path": fixture_path.as_posix(),
        "fixture_sha256": _normalized_sha256(fixture_path),
        "optimizer_state_hashes": [
            {
                "D": record["D"],
                "lineage": record["lineage"],
                "seed": record["seed"],
                "state_sha256": record["state_sha256"],
            }
            for record in optimizer_records
        ],
        "choices": choices,
    }


def _load_comparators_after_ledger() -> dict[str, Any]:
    try:
        module = importlib.import_module("scripts.phase44_reference_comparators")
    except ModuleNotFoundError:
        module = importlib.import_module("phase44_reference_comparators")
    return module.load_frozen_comparators()


def _evaluation_gate(record: dict[str, Any]) -> dict[str, bool]:
    gates = {
        "completed": record["completed"],
        "acceptance": 0.15 <= record["acceptance_rate"] <= 0.85,
        "rhat": record["rhat"] <= 1.05,
        "effective_sample_size": record["effective_sample_size"] >= 50000,
        "standard_error": record["energy_standard_error"] <= 2.5e-4,
        "variance": record["energy_variance"] <= 0.02,
        "antisymmetry": record["symmetry"]["antisymmetry_residual"] <= 1e-12,
        "correlator_symmetry": record["symmetry"][
            "correlator_symmetry_residual"
        ]
        <= 1e-12,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _optimizer_gate(record: dict[str, Any]) -> dict[str, bool]:
    finite_history = all(
        all(
            math.isfinite(point[field])
            for field in (
                "energy",
                "energy_variance",
                "energy_standard_error",
                "acceptance_rate",
                "effective_sample_size",
                "rhat",
                "gradient_norm",
                "gradient_clip_scale",
            )
        )
        for point in record["history"]
    )
    acceptance = all(
        0.15 <= point["acceptance_rate"] <= 0.85 for point in record["history"]
    )
    symmetry = all(
        point["antisymmetry_residual"] <= 1e-12
        and point["correlator_symmetry_residual"] <= 1e-12
        for point in record["history"]
    )
    amplitudes = torch.tensor(record["amplitudes"], dtype=torch.float64)
    gates = {
        "completed": record["completed"] and record["completed_steps"] == 100,
        "finite_history": finite_history,
        "acceptance": acceptance,
        "symmetry": symmetry,
        "amplitude_not_at_bound": bool(torch.all(torch.abs(amplitudes) < 1.0)),
        "no_forbidden_materialization": record["materialization"]
        == {
            "D_to_the_N_tensor": False,
            "full_alternating_coefficient_tensor": False,
            "virtual_paths": 0,
        },
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _combine_confirmation(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 2:
        raise ValueError("exactly two confirmation records are required")
    standard_errors = [record["energy_standard_error"] for record in records]
    weights = [1.0 / error**2 for error in standard_errors]
    energy = sum(weight * record["energy"] for weight, record in zip(weights, records)) / sum(weights)
    standard_error = math.sqrt(1.0 / sum(weights))
    difference = abs(records[0]["energy"] - records[1]["energy"])
    allowance = 5.0 * math.sqrt(standard_errors[0] ** 2 + standard_errors[1] ** 2) + 2e-4
    return {
        "inverse_variance_energy": energy,
        "inverse_variance_standard_error": standard_error,
        "absolute_seed_difference": difference,
        "seed_agreement_allowance": allowance,
        "seed_agreement_pass": difference <= allowance,
    }


def run(
    fixture_path: Path,
    ledger_path: Path,
    output_path: Path,
    archive_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("Phase 44 final output already exists")
    archive_paths = {
        basis_order: archive_dir / f"phase44_n4_d{basis_order}_samples.npz"
        for basis_order in D_AXIS
    }
    existing_archives = [path for path in archive_paths.values() if path.exists()]
    if existing_archives:
        raise FileExistsError(f"Phase 44 sample archive already exists: {existing_archives[0]}")

    fixture, source_carrier = _load_reference_free_fixture(fixture_path)
    exponents = torch.tensor(EXPONENTS, dtype=torch.float64)
    initial_amplitudes = torch.zeros(len(EXPONENTS), dtype=torch.float64)
    optimizer_results: dict[tuple[int, int], dict[str, Any]] = {}
    optimizer_records: list[dict[str, Any]] = []
    resume_clean_pass = False
    for basis_order in D_AXIS:
        raw = _initial_carrier(source_carrier, basis_order)
        for lineage, seed in enumerate(OPTIMIZER_SEEDS[basis_order], start=1):
            force_partial = basis_order == 6 and lineage == 1
            result, record = _run_optimizer(
                basis_order,
                lineage,
                seed,
                raw,
                initial_amplitudes,
                exponents,
                checkpoint_dir,
                force_partial=force_partial,
            )
            optimizer_results[(basis_order, lineage)] = result
            record["gates"] = _optimizer_gate(record)
            optimizer_records.append(record)
            print(
                f"optimized D={basis_order} lineage={lineage} "
                f"E_last={record['history'][-1]['energy']:.10f}",
                flush=True,
            )
            if force_partial:
                clean, clean_record = _run_optimizer(
                    basis_order,
                    lineage,
                    seed,
                    raw,
                    initial_amplitudes,
                    exponents,
                    checkpoint_dir,
                    clean_suffix="clean",
                )
                resume_clean_pass = _resume_clean_equal(result, clean)
                record["forced_resume_clean_comparison"] = {
                    "clean_checkpoint_path": clean_record["checkpoint_path"],
                    "bitwise_trajectory_pass": resume_clean_pass,
                    "clean_elapsed_seconds": clean_record["elapsed_seconds"],
                    "clean_peak_cpu_rss_bytes": clean_record["peak_cpu_rss_bytes"],
                }

    selection_results: dict[tuple[int, int], dict[str, Any]] = {}
    selection_records: list[dict[str, Any]] = []
    for basis_order in D_AXIS:
        for lineage, seed in enumerate(SELECTION_SEEDS[basis_order], start=1):
            state = optimizer_results[(basis_order, lineage)]
            config = _selection_config(seed)
            result, record = _run_evaluation(
                config,
                state["orbitals"],
                state["amplitudes"],
                exponents,
                checkpoint_dir / f"d{basis_order}_lineage{lineage}_selection.pt",
            )
            record.update(
                {
                    "D": basis_order,
                    "lineage": lineage,
                    "seed": seed,
                    "optimizer_seed": OPTIMIZER_SEEDS[basis_order][lineage - 1],
                    "state_sha256": _tensor_sha256(
                        state["orbitals"], state["amplitudes"], exponents
                    ),
                }
            )
            record["gates"] = _evaluation_gate(record)
            selection_results[(basis_order, lineage)] = result
            selection_records.append(record)
            print(
                f"selection D={basis_order} lineage={lineage} "
                f"E={record['energy']:.10f} se={record['energy_standard_error']:.2e}",
                flush=True,
            )

    ledger = _selection_ledger(fixture_path, optimizer_records, selection_records)
    if ledger_path.exists():
        existing_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if existing_ledger != ledger:
            raise AssertionError("pre-reference selection ledger changed")
    else:
        _write(ledger_path, ledger)
    ledger_sha256 = _normalized_sha256(ledger_path)

    # Reference/comparator artifacts are opened only after the immutable
    # selection ledger above exists and has been hashed.
    if any(
        name in sys.modules
        for name in (
            "scripts.phase44_reference_comparators",
            "phase44_reference_comparators",
        )
    ):
        raise AssertionError("reference comparator module was imported before selection")
    comparators = _load_comparators_after_ledger()
    choices = {choice["D"]: choice for choice in ledger["choices"]}
    confirmation_records: list[dict[str, Any]] = []
    confirmation_samples: dict[tuple[int, int], torch.Tensor] = {}
    for basis_order in D_AXIS:
        lineage = choices[basis_order]["selected_lineage"]
        state = optimizer_results[(basis_order, lineage)]
        for confirmation_index, seed in enumerate(
            CONFIRMATION_SEEDS[basis_order], start=1
        ):
            config = _confirmation_config(seed)
            result, record = _run_evaluation(
                config,
                state["orbitals"],
                state["amplitudes"],
                exponents,
                checkpoint_dir
                / f"d{basis_order}_selected_lineage{lineage}_confirmation{confirmation_index}.pt",
            )
            record.update(
                {
                    "D": basis_order,
                    "lineage": lineage,
                    "confirmation_index": confirmation_index,
                    "seed": seed,
                    "state_sha256": choices[basis_order]["state_sha256"],
                }
            )
            record["gates"] = _evaluation_gate(record)
            confirmation_records.append(record)
            confirmation_samples[(basis_order, confirmation_index)] = result["samples"]
            print(
                f"confirmation D={basis_order} seed={seed} "
                f"E={record['energy']:.10f} se={record['energy_standard_error']:.2e}",
                flush=True,
            )

    archive_records = {}
    archive_dir.mkdir(parents=True, exist_ok=True)
    for basis_order, path in archive_paths.items():
        arrays = {}
        for lineage in (1, 2):
            arrays[f"selection_lineage{lineage}"] = (
                selection_results[(basis_order, lineage)]["samples"]
                .detach()
                .cpu()
                .numpy()
            )
        for confirmation_index in (1, 2):
            arrays[f"confirmation{confirmation_index}"] = (
                confirmation_samples[(basis_order, confirmation_index)]
                .detach()
                .cpu()
                .numpy()
            )
        np.savez_compressed(path, **arrays)
        archive_records[str(basis_order)] = {
            "path": path.as_posix(),
            "sha256": _raw_sha256(path),
            "arrays": {key: list(value.shape) for key, value in arrays.items()},
        }

    combined = {}
    for basis_order in D_AXIS:
        records = [
            record
            for record in confirmation_records
            if record["D"] == basis_order
        ]
        combined[str(basis_order)] = _combine_confirmation(records)
    monotonic_gates = []
    for lower, upper in zip(D_AXIS, D_AXIS[1:]):
        lower_record = combined[str(lower)]
        upper_record = combined[str(upper)]
        allowance = (
            5.0
            * math.sqrt(
                lower_record["inverse_variance_standard_error"] ** 2
                + upper_record["inverse_variance_standard_error"] ** 2
            )
            + 2e-4
        )
        monotonic_gates.append(
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
    point_gates = []
    for basis_order in D_AXIS:
        record = combined[str(basis_order)]
        error = abs(
            record["inverse_variance_energy"]
            - comparators["d14_numerical_reference"]
        )
        conservative_error = error + 5.0 * record["inverse_variance_standard_error"]
        noci_error = comparators["noci_absolute_reference_errors"][str(basis_order)]
        point_gates.append(
            {
                "D": basis_order,
                "explicit_absolute_reference_error": error,
                "explicit_conservative_error": conservative_error,
                "noci_absolute_reference_error": noci_error,
                "conservative_error_ratio": conservative_error / noci_error,
                "pass": conservative_error <= 0.5 * noci_error,
            }
        )
    consecutive_pairs = [
        [first["D"], second["D"]]
        for first, second in zip(point_gates, point_gates[1:])
        if first["pass"] and second["pass"]
    ]
    acceptance = {
        "all_optimizer_gates_pass": all(
            record["gates"]["all_pass"] for record in optimizer_records
        ),
        "forced_resume_clean_pass": resume_clean_pass,
        "all_selection_gates_pass": all(
            record["gates"]["all_pass"] for record in selection_records
        ),
        "all_confirmation_gates_pass": all(
            record["gates"]["all_pass"] for record in confirmation_records
        ),
        "all_confirmation_seed_agreements_pass": all(
            record["seed_agreement_pass"] for record in combined.values()
        ),
        "D_monotonicity_pass": all(record["pass"] for record in monotonic_gates),
        "two_consecutive_D_advantage_pass": bool(consecutive_pairs),
    }
    acceptance["phase44_interacting_d_gate_pass"] = all(acceptance.values())
    artifact = {
        "schema_version": 1,
        "experiment": "phase44_n4_explicit_correlation_d_gate",
        "evidence_level": "preregistered internal numerical evidence",
        "scientific_boundary": "interacting N4 carrier-basis differentiator; not external replication, a new Jastrow ansatz, a scalability/superiority claim, or authorization for Paper B",
        "adr": "docs/decisions/0033-preregister-n4-explicit-correlation-d-gate.md",
        "reference_firewall": {
            "fixture_loaded_before_selection": fixture_path.as_posix(),
            "fixture_disclosure": fixture["disclosure"],
            "selection_ledger": ledger_path.as_posix(),
            "selection_ledger_sha256": ledger_sha256,
            "comparator_artifacts_opened_only_after_ledger_hash": True,
        },
        "frozen_axes": {"N": 4, "D": list(D_AXIS), "chi": 1, "P": 5},
        "frozen_exponents": list(EXPONENTS),
        "optimizer_configs": {
            str(seed): asdict(_optimizer_config(seed))
            for seeds in OPTIMIZER_SEEDS.values()
            for seed in seeds
        },
        "optimizer_runs": optimizer_records,
        "selection_runs": selection_records,
        "selection_choices": ledger["choices"],
        "comparators": comparators,
        "confirmation_runs": confirmation_records,
        "combined_confirmations": combined,
        "D_monotonicity": monotonic_gates,
        "point_advantage_gates": point_gates,
        "consecutive_advantage_pairs": consecutive_pairs,
        "sample_archives": archive_records,
        "source_hashes": {
            path.as_posix(): _normalized_sha256(path)
            for path in (
                Path("src/femps/algorithms/correlated_exterior_vmc.py"),
                Path("src/femps/algorithms/correlated_exterior_vmc_training.py"),
                Path("scripts/benchmark_phase44_n4_explicit_correlation_d_gate.py"),
                Path("scripts/phase44_reference_comparators.py"),
                Path("docs/decisions/0033-preregister-n4-explicit-correlation-d-gate.md"),
                fixture_path,
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
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    arguments = parser.parse_args()
    result = run(
        arguments.fixture,
        arguments.ledger,
        arguments.output,
        arguments.archive_dir,
        arguments.checkpoint_dir,
    )
    print(json.dumps(result["acceptance"], indent=2))


if __name__ == "__main__":
    main()
