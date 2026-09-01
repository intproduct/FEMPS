"""Run the preregistered restored Phase 39 N4,D8 internal calculation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from femps.algorithms import (  # noqa: E402
    load_slater_source_command_config,
    validate_slater_source_result,
)
from scripts.benchmark_phase28_soft_coulomb_transferability import (  # noqa: E402
    _dense_ci,
)
from scripts.benchmark_phase38_clean_source_seed_robustness import (  # noqa: E402
    _canonical_json_sha256,
    _candidates,
    _command_record,
    _energies,
    _optimizer_failure_count,
    _stage_orbitals,
    _text_sha256,
    _write,
)


def run(
    config_path: Path,
    checkpoint_dir: Path,
    phase37_artifact: Path,
    output: Path,
) -> dict:
    if checkpoint_dir.exists() or output.exists():
        raise ValueError("registered Phase 39 D8 checkpoint/output already exists")
    config, record = load_slater_source_command_config(config_path)
    if (config.particles, config.basis_order, config.max_terms) != (4, 8, 4):
        raise ValueError("restored Phase 39 requires exactly N4,D8,K4")
    thresholds = record["acceptance"]
    checkpoint_dir.mkdir(parents=True)

    checkpoints = {
        "resumed": checkpoint_dir / "resumed.pt",
        "clean": checkpoint_dir / "clean.pt",
    }
    raw_outputs = {
        "resumed": checkpoint_dir / "resumed.json",
        "clean": checkpoint_dir / "clean.json",
    }
    commands: list[list[str]] = []

    command, partial_call = _command_record(
        config_path,
        checkpoints["resumed"],
        raw_outputs["resumed"],
        maximum_adaptive_stages=1,
    )
    commands.append(command)
    partial = partial_call["result"]
    if partial["completed"] or partial["current_terms"] != 2:
        raise RuntimeError("registered interruption did not stop after K2")
    partial_stage_sha256 = _canonical_json_sha256(partial["stages"])

    command, resumed_call = _command_record(
        config_path,
        checkpoints["resumed"],
        raw_outputs["resumed"],
        resume=True,
    )
    commands.append(command)
    command, clean_call = _command_record(
        config_path,
        checkpoints["clean"],
        raw_outputs["clean"],
    )
    commands.append(command)
    resumed = resumed_call["result"]
    clean = clean_call["result"]
    results = {"resumed": resumed, "clean": clean}
    for result in results.values():
        validate_slater_source_result(result, require_completed=True)

    # Open truth and the historical D6 control only after both D8 lineages are frozen.
    dense_truth = _dense_ci(config.basis_order)
    phase37 = json.loads(phase37_artifact.read_text(encoding="utf-8"))
    if not phase37["acceptance"]["phase37_slater_source_solver_pass"]:
        raise RuntimeError("registered D6 historical control is not accepted")
    phase37_energy = phase37["comparison"]["resumed_energies"][-1]

    energies = {name: _energies(result) for name, result in results.items()}
    candidates = {name: _candidates(result) for name, result in results.items()}
    differences = [
        abs(left - right)
        for left, right in zip(
            energies["resumed"], energies["clean"], strict=True
        )
    ]
    ci_energy = dense_truth["energy"]
    source_errors = [values[0] - ci_energy for values in energies.values()]
    final_errors = [values[-1] - ci_energy for values in energies.values()]
    source_variances = [
        result["stages"][0]["optimizer_result"]["energy_variance"]
        for result in results.values()
    ]
    final_variances = [
        result["stages"][-1]["optimizer_result"]["energy_variance"]
        for result in results.values()
    ]
    stage_points = [
        stage["optimizer_result"]
        for result in results.values()
        for stage in result["stages"]
    ]
    failures = _optimizer_failure_count(list(results.values()))
    command_times = {
        "resumed_total": (
            partial["total_elapsed_seconds_this_call"]
            + resumed["total_elapsed_seconds_this_call"]
        ),
        "clean": clean["total_elapsed_seconds_this_call"],
    }
    acceptance = {
        "both_complete_pass": all(result["completed"] for result in results.values()),
        "registered_interruption_pass": (
            not partial["completed"]
            and partial["current_terms"] == 2
            and resumed["resumed"]
            and partial_stage_sha256
            == _canonical_json_sha256(resumed["stages"][:2])
        ),
        "clean_source_boundary_pass": all(
            not result["source_construction"]["historical_checkpoint_used"]
            and not result["source_construction"]["ci_initializer_used"]
            for result in results.values()
        ),
        "candidate_reproduction_pass": candidates["resumed"] == candidates["clean"],
        "clean_resume_energy_match_pass": max(differences)
        <= thresholds["resume_energy_tolerance"],
        "energy_nonincreasing_pass": all(
            right <= left + thresholds["energy_nesting_tolerance"]
            for lineage in energies.values()
            for left, right in zip(lineage, lineage[1:])
        ),
        "source_accuracy_pass": all(
            abs(error) <= thresholds["source_ci_error_maximum"]
            and variance <= thresholds["source_variance_maximum"]
            for error, variance in zip(
                source_errors, source_variances, strict=True
            )
        ),
        "final_accuracy_pass": all(
            abs(error) <= thresholds["final_ci_error_maximum"]
            and variance <= thresholds["final_variance_maximum"]
            for error, variance in zip(final_errors, final_variances, strict=True)
        ),
        "optimizer_failure_count_pass": failures
        <= thresholds["optimizer_failure_count_maximum"],
        "stage_scientific_records_pass": all(
            point["norm_error"] <= thresholds["norm_error_maximum"]
            and point["structural_antisymmetry_residual"]
            <= thresholds["antisymmetry_residual_maximum"]
            and point["materialized_antisymmetry_residual"]
            <= thresholds["antisymmetry_residual_maximum"]
            and point["structural_counts"]["enumerated_virtual_paths"] == 0
            and point["structural_counts"]["materialized_particle_coefficients"]
            == 0
            and point["total_elapsed_seconds_this_call"]
            <= thresholds["stage_wall_time_maximum_seconds"]
            and point["peak_cpu_rss_bytes"]
            <= thresholds["peak_cpu_rss_maximum_bytes"]
            for point in stage_points
        ),
        "operator_factorization_pass": all(
            result["operator_metadata"]["dense_relative_factorization_error"]
            <= thresholds["factorization_error_maximum"]
            for result in results.values()
        ),
        "command_resource_pass": (
            max(command_times.values())
            <= thresholds["command_wall_time_maximum_seconds"]
            and max(
                partial["peak_cpu_rss_bytes"],
                *(result["peak_cpu_rss_bytes"] for result in results.values()),
            )
            <= thresholds["peak_cpu_rss_maximum_bytes"]
        ),
        "external_cap_boundary_pass": all(
            result["automatic_stopping_rule"] == "not_admitted"
            and result["external_max_terms_required"]
            for result in results.values()
        ),
    }
    acceptance["phase39_n4_d8_internal_pass"] = all(acceptance.values())

    source_files = (
        "docs/decisions/0031-preregister-restored-phase39-n4-d8.md",
        "docs/experiments/configs/phase39_n4_d8_k4.json",
        "src/femps/algorithms/slater_source_contract.py",
        "src/femps/algorithms/slater_source_training.py",
        "scripts/run_femps_slater_source_solver.py",
        "scripts/benchmark_phase38_clean_source_seed_robustness.py",
        "scripts/benchmark_phase39_n4_d8_clean_source.py",
        "scripts/verify_phase39_n4_d8_clean_source.py",
    )
    artifact = {
        "schema_version": 1,
        "experiment": "phase39_n4_d8_clean_source_internal",
        "evidence_level": "internal numerical evidence",
        "scientific_boundary": (
            "single restored N4,D8 clean-source NOCI-equivalent calculation; "
            "not a new ansatz, NOCI advantage, continuum, or scaling claim"
        ),
        "registered_config": record,
        "source_hash_policy": "UTF-8 text normalized to LF before SHA-256",
        "source_hashes": {path: _text_sha256(Path(path)) for path in source_files},
        "execution": {
            "public_commands": commands,
            "truth_opened_after_both_d8_optimizations": True,
            "outcome_dependent_retries": 0,
            "additional_small_points_authorized": 0,
            "partial": {
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
        "results": results,
        "stage_orbitals": {
            name: _stage_orbitals(checkpoint, 4)
            for name, checkpoint in checkpoints.items()
        },
        "dense_ci_comparator": dense_truth,
        "historical_d6_control": {
            "path": str(phase37_artifact),
            "normalized_text_sha256": _text_sha256(phase37_artifact),
            "basis_order": 6,
            "final_energy": phase37_energy,
            "initialization": (
                "optimized K1 canonical Slater followed by preregistered "
                "adaptive K2-K4 growth"
            ),
        },
        "comparison": {
            "energies": energies,
            "selected_candidates": candidates,
            "clean_resume_energy_absolute_differences": differences,
            "source_errors_vs_dense_ci": dict(
                zip(results, source_errors, strict=True)
            ),
            "final_errors_vs_dense_ci": dict(
                zip(results, final_errors, strict=True)
            ),
            "source_variances": dict(zip(results, source_variances, strict=True)),
            "final_variances": dict(zip(results, final_variances, strict=True)),
            "d6_to_d8_final_energy_change": energies["resumed"][-1]
            - phase37_energy,
            "optimizer_failure_count": failures,
            "ordinary_particle_tt_ranks": {
                name: [
                    stage["optimizer_result"]["ordinary_particle_tt_ranks"]
                    for stage in result["stages"]
                ]
                for name, result in results.items()
            },
            "femps_correlation_multiplicity": [1, 2, 3, 4],
        },
        "acceptance": acceptance,
        "environment": {
            "python": platform.python_version(),
            "device": "cpu",
        },
    }
    _write(output, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/experiments/configs/phase39_n4_d8_k4.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase39_n4_d8_clean_source"),
    )
    parser.add_argument(
        "--phase37-artifact",
        type=Path,
        default=Path("docs/experiments/results/phase37_slater_source_solver.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/phase39_n4_d8_clean_source.json"),
    )
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.checkpoint_dir,
        arguments.phase37_artifact,
        arguments.output,
    )
    print(
        json.dumps(
            {
                "accepted": result["acceptance"]["phase39_n4_d8_internal_pass"],
                "energies": result["comparison"]["energies"],
                "selected_candidates": result["comparison"][
                    "selected_candidates"
                ],
                "final_errors_vs_dense_ci": result["comparison"][
                    "final_errors_vs_dense_ci"
                ],
                "output": str(arguments.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
