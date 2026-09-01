"""Run the preregistered Phase 38 clean-source seed robustness audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from femps.algorithms import (  # noqa: E402
    canonical_slater_orbitals,
    load_diagonal_path_checkpoint,
    load_slater_source_command_config,
    validate_slater_source_result,
)
from scripts.benchmark_phase28_soft_coulomb_transferability import (  # noqa: E402
    _dense_ci,
)


PUBLIC_COMMAND = Path("scripts/run_femps_slater_source_solver.py")


def _text_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _complex_tensor(tensor: torch.Tensor) -> list:
    return torch.view_as_real(
        tensor.detach().to(dtype=torch.complex128, device="cpu").contiguous()
    ).tolist()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _child(checkpoint: Path, label: str) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}.{label}.pt")


def _optimizer_checkpoint(adaptive: Path, terms: int) -> Path:
    return adaptive.with_name(f"{adaptive.stem}.K{terms}.optimizer.pt")


def _stage_orbitals(checkpoint: Path, max_terms: int) -> list[dict]:
    source = canonical_slater_orbitals(
        load_diagonal_path_checkpoint(_child(checkpoint, "source.optimizer"))[
            "best_raw"
        ]
    )
    adaptive = _child(checkpoint, "adaptive")
    records = [{"terms": 1, "values": _complex_tensor(source)}]
    for terms in range(2, max_terms + 1):
        orbitals = canonical_slater_orbitals(
            load_diagonal_path_checkpoint(
                _optimizer_checkpoint(adaptive, terms)
            )["best_raw"]
        )
        records.append({"terms": terms, "values": _complex_tensor(orbitals)})
    return records


def _nonseed_record(record: dict) -> dict:
    sanitized = copy.deepcopy(record)
    sanitized["source_optimizer"]["seed"] = "<source-seed>"
    for stage in sanitized["adaptive"]["stages"]:
        stage["candidate_seed"] = "<candidate-seed>"
        stage["optimizer_seed"] = "<optimizer-seed>"
    sanitized["checkpoint_path"] = "<checkpoint-path>"
    sanitized["output_path"] = "<output-path>"
    return sanitized


def _command_record(
    config: Path,
    checkpoint: Path,
    output: Path,
    *,
    resume: bool = False,
    maximum_adaptive_stages: int | None = None,
) -> tuple[list[str], dict]:
    arguments = [
        sys.executable,
        str(PUBLIC_COMMAND),
        "--config",
        str(config),
        "--max-k",
        "4",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    ]
    if resume:
        arguments.append("--resume")
    if maximum_adaptive_stages is not None:
        arguments.extend(
            ["--max-adaptive-stages-this-call", str(maximum_adaptive_stages)]
        )
    completed = subprocess.run(
        arguments,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    validate_slater_source_result(result)
    portable = ["python", *arguments[1:]]
    return portable, {
        "stdout": json.loads(completed.stdout),
        "result": result,
    }


def _energies(result: dict) -> list[float]:
    return [stage["optimizer_result"]["energy"] for stage in result["stages"]]


def _candidates(result: dict) -> list[int]:
    return [stage["selected_candidate"] for stage in result["stages"][1:]]


def _optimizer_failure_count(results: list[dict]) -> int:
    return sum(
        not point["completed"] or not point["refinement"]["accepted"]
        for result in results
        for point in (stage["optimizer_result"] for stage in result["stages"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-a",
        type=Path,
        default=Path("docs/experiments/configs/phase38_n4_d6_k4_seed_a.json"),
    )
    parser.add_argument(
        "--config-b",
        type=Path,
        default=Path("docs/experiments/configs/phase38_n4_d6_k4_seed_b.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase38_clean_source_seed_robustness"),
    )
    parser.add_argument(
        "--phase37-artifact",
        type=Path,
        default=Path("docs/experiments/results/phase37_slater_source_solver.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase38_clean_source_seed_robustness.json"
        ),
    )
    args = parser.parse_args()

    if args.checkpoint_dir.exists() or args.output.exists():
        raise ValueError("registered Phase 38 checkpoint/output already exists")
    config_a, record_a = load_slater_source_command_config(args.config_a)
    config_b, record_b = load_slater_source_command_config(args.config_b)
    if _nonseed_record(record_a) != _nonseed_record(record_b):
        raise ValueError("Phase 38 configurations differ outside seeds and paths")
    thresholds = record_a["acceptance"]
    args.checkpoint_dir.mkdir(parents=True)

    checkpoints = {
        "a_resumed": args.checkpoint_dir / "seed_a_resumed.pt",
        "a_clean": args.checkpoint_dir / "seed_a_clean.pt",
        "b_clean": args.checkpoint_dir / "seed_b_clean.pt",
    }
    raw_outputs = {
        "a_resumed": args.checkpoint_dir / "seed_a_resumed.json",
        "a_clean": args.checkpoint_dir / "seed_a_clean.json",
        "b_clean": args.checkpoint_dir / "seed_b_clean.json",
    }
    commands = []

    command, partial_call = _command_record(
        args.config_a,
        checkpoints["a_resumed"],
        raw_outputs["a_resumed"],
        maximum_adaptive_stages=1,
    )
    commands.append(command)
    partial = partial_call["result"]
    if partial["completed"] or partial["current_terms"] != 2:
        raise RuntimeError("registered Phase 38 interruption did not stop at K2")
    partial_stage_sha256 = _canonical_json_sha256(partial["stages"])

    command, a_resumed_call = _command_record(
        args.config_a,
        checkpoints["a_resumed"],
        raw_outputs["a_resumed"],
        resume=True,
    )
    commands.append(command)
    command, a_clean_call = _command_record(
        args.config_a,
        checkpoints["a_clean"],
        raw_outputs["a_clean"],
    )
    commands.append(command)
    command, b_clean_call = _command_record(
        args.config_b,
        checkpoints["b_clean"],
        raw_outputs["b_clean"],
    )
    commands.append(command)

    a_resumed = a_resumed_call["result"]
    a_clean = a_clean_call["result"]
    b_clean = b_clean_call["result"]
    results = [a_resumed, a_clean, b_clean]
    for result in results:
        validate_slater_source_result(result, require_completed=True)

    # No truth or historical result is opened until all fresh runs are frozen.
    dense_truth = _dense_ci(config_a.basis_order)
    phase37 = json.loads(args.phase37_artifact.read_text(encoding="utf-8"))
    if not phase37["acceptance"]["phase37_slater_source_solver_pass"]:
        raise RuntimeError("Phase 37 historical control is not accepted")
    phase37_final = phase37["comparison"]["resumed_energies"][-1]
    ci_energy = dense_truth["energy"]

    energies = {name: _energies(result) for name, result in zip(
        ("a_resumed", "a_clean", "b_clean"), results, strict=True
    )}
    candidates = {name: _candidates(result) for name, result in zip(
        ("a_resumed", "a_clean", "b_clean"), results, strict=True
    )}
    fresh_primary = [a_resumed, b_clean]
    primary_names = ("a_resumed", "b_clean")
    fresh_final_energies = [energies[name][-1] for name in primary_names]
    combined_final_energies = [phase37_final, *fresh_final_energies]
    final_errors = [energy - ci_energy for energy in fresh_final_energies]
    final_variances = [
        result["stages"][-1]["optimizer_result"]["energy_variance"]
        for result in fresh_primary
    ]
    source_errors = [energies[name][0] - ci_energy for name in primary_names]
    source_variances = [
        result["stages"][0]["optimizer_result"]["energy_variance"]
        for result in fresh_primary
    ]
    a_differences = [
        abs(left - right)
        for left, right in zip(
            energies["a_resumed"], energies["a_clean"], strict=True
        )
    ]
    stage_points = [
        stage["optimizer_result"] for result in results for stage in result["stages"]
    ]
    optimizer_failures = _optimizer_failure_count(results)
    final_spread = max(combined_final_energies) - min(combined_final_energies)
    fresh_phase37_differences = [
        abs(energy - phase37_final) for energy in fresh_final_energies
    ]
    structural_pass = all(
        point["norm_error"] <= thresholds["norm_error_maximum"]
        and point["structural_antisymmetry_residual"]
        <= thresholds["antisymmetry_residual_maximum"]
        and point["materialized_antisymmetry_residual"]
        <= thresholds["antisymmetry_residual_maximum"]
        and point["structural_counts"]["enumerated_virtual_paths"] == 0
        and point["structural_counts"]["materialized_particle_coefficients"] == 0
        and point["total_elapsed_seconds_this_call"]
        <= thresholds["stage_wall_time_maximum_seconds"]
        and point["peak_cpu_rss_bytes"] <= thresholds["peak_cpu_rss_maximum_bytes"]
        for point in stage_points
    )
    command_times = {
        "a_resumed_total": (
            partial["total_elapsed_seconds_this_call"]
            + a_resumed["total_elapsed_seconds_this_call"]
        ),
        "a_clean": a_clean["total_elapsed_seconds_this_call"],
        "b_clean": b_clean["total_elapsed_seconds_this_call"],
    }
    acceptance = {
        "all_complete_pass": all(result["completed"] for result in results),
        "registered_interruption_pass": (
            not partial["completed"]
            and partial["current_terms"] == 2
            and a_resumed["resumed"]
            and partial["stages"] == a_resumed["stages"][:2]
        ),
        "clean_source_boundary_pass": all(
            not result["source_construction"]["historical_checkpoint_used"]
            and not result["source_construction"]["ci_initializer_used"]
            for result in results
        ),
        "a_candidate_reproduction_pass": (
            candidates["a_resumed"] == candidates["a_clean"]
        ),
        "a_clean_resume_energy_match_pass": (
            max(a_differences) <= thresholds["resume_energy_tolerance"]
        ),
        "energy_nonincreasing_pass": all(
            right <= left + thresholds["energy_nesting_tolerance"]
            for lineage in energies.values()
            for left, right in zip(lineage, lineage[1:])
        ),
        "source_accuracy_pass": all(
            abs(error) <= thresholds["source_ci_error_maximum"]
            and variance <= thresholds["source_variance_maximum"]
            for error, variance in zip(source_errors, source_variances, strict=True)
        ),
        "final_accuracy_pass": all(
            abs(error) <= thresholds["final_ci_error_maximum"]
            and variance <= thresholds["final_variance_maximum"]
            for error, variance in zip(final_errors, final_variances, strict=True)
        ),
        "final_energy_spread_pass": (
            final_spread <= thresholds["final_energy_spread_maximum"]
        ),
        "phase37_difference_pass": (
            max(fresh_phase37_differences)
            <= thresholds["fresh_difference_from_phase37_maximum"]
        ),
        "optimizer_failure_count_pass": (
            optimizer_failures <= thresholds["optimizer_failure_count_maximum"]
        ),
        "stage_scientific_records_pass": structural_pass,
        "operator_factorization_pass": all(
            result["operator_metadata"]["dense_relative_factorization_error"]
            <= thresholds["factorization_error_maximum"]
            for result in results
        ),
        "command_resource_pass": (
            max(command_times.values())
            <= thresholds["command_wall_time_maximum_seconds"]
            and max(result["peak_cpu_rss_bytes"] for result in [partial, *results])
            <= thresholds["peak_cpu_rss_maximum_bytes"]
        ),
        "external_cap_boundary_pass": all(
            result["automatic_stopping_rule"] == "not_admitted"
            and result["external_max_terms_required"]
            for result in results
        ),
    }
    acceptance["phase38_clean_source_seed_robustness_pass"] = all(
        acceptance.values()
    )

    source_files = (
        "docs/decisions/0027-preregister-clean-source-seed-robustness.md",
        "docs/experiments/configs/phase38_n4_d6_k4_seed_a.json",
        "docs/experiments/configs/phase38_n4_d6_k4_seed_b.json",
        "src/femps/algorithms/slater_source_contract.py",
        "src/femps/algorithms/slater_source_training.py",
        "scripts/run_femps_slater_source_solver.py",
        "scripts/benchmark_phase38_clean_source_seed_robustness.py",
        "scripts/verify_phase38_clean_source_seed_robustness.py",
    )
    artifact = {
        "schema_version": 1,
        "experiment": "phase38_clean_source_seed_robustness",
        "evidence_level": "numerical",
        "scientific_boundary": (
            "two additional registered clean-source schedules at N4,D6; "
            "no universal seed-independence, scaling, or automatic-stop claim"
        ),
        "registered_configs": {"a": record_a, "b": record_b},
        "source_hash_policy": "UTF-8 text normalized to LF before SHA-256",
        "source_hashes": {path: _text_sha256(Path(path)) for path in source_files},
        "phase37_control": {
            "path": str(args.phase37_artifact),
            "normalized_text_sha256": _text_sha256(args.phase37_artifact),
            "final_energy": phase37_final,
        },
        "execution": {
            "public_commands": commands,
            "truth_opened_after_all_fresh_optimizations": True,
            "outcome_dependent_retries": 0,
            "partial_a": {
                "completed": partial["completed"],
                "current_terms": partial["current_terms"],
                "stage_sha256": partial_stage_sha256,
                "energies": _energies(partial),
                "selected_candidates": _candidates(partial),
                "total_elapsed_seconds_this_call": partial[
                    "total_elapsed_seconds_this_call"
                ],
                "peak_cpu_rss_bytes": partial["peak_cpu_rss_bytes"],
            },
            "command_times_seconds": command_times,
        },
        "results": {
            "a_resumed": a_resumed,
            "a_clean": a_clean,
            "b_clean": b_clean,
        },
        "stage_orbitals": {
            name: _stage_orbitals(checkpoints[name], 4) for name in checkpoints
        },
        "dense_ci_comparator": dense_truth,
        "comparison": {
            "energies": energies,
            "selected_candidates": candidates,
            "a_clean_resume_energy_absolute_differences": a_differences,
            "fresh_final_errors_vs_dense_ci": dict(zip(primary_names, final_errors)),
            "fresh_final_variances": dict(zip(primary_names, final_variances)),
            "fresh_source_errors_vs_dense_ci": dict(zip(primary_names, source_errors)),
            "fresh_source_variances": dict(zip(primary_names, source_variances)),
            "combined_phase37_a_b_final_energies": combined_final_energies,
            "combined_final_energy_spread": final_spread,
            "fresh_absolute_differences_from_phase37": dict(
                zip(primary_names, fresh_phase37_differences)
            ),
            "optimizer_failure_count": optimizer_failures,
            "ordinary_particle_tt_ranks": {
                name: [
                    stage["optimizer_result"]["ordinary_particle_tt_ranks"]
                    for stage in result["stages"]
                ]
                for name, result in zip(
                    ("a_resumed", "a_clean", "b_clean"), results, strict=True
                )
            },
            "femps_correlation_multiplicity": [1, 2, 3, 4],
        },
        "acceptance": acceptance,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
    }
    _write(args.output, artifact)
    print(
        json.dumps(
            {
                "accepted": acceptance[
                    "phase38_clean_source_seed_robustness_pass"
                ],
                "energies": energies,
                "selected_candidates": candidates,
                "combined_final_energy_spread": final_spread,
                "maximum_final_error_vs_dense_ci": max(map(abs, final_errors)),
                "maximum_final_variance": max(final_variances),
                "optimizer_failure_count": optimizer_failures,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
