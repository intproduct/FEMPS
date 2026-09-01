"""Run the preregistered Phase 37 clean Slater-source FEMPS closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from femps.algorithms import (
    canonical_slater_orbitals,
    load_diagonal_path_checkpoint,
    load_slater_source_command_config,
    run_slater_source_adaptive_solver,
    validate_slater_source_result,
)

from scripts.benchmark_phase28_soft_coulomb_transferability import _dense_ci


def _text_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _complex_tensor(tensor: torch.Tensor) -> list:
    return torch.view_as_real(
        tensor.detach().to(dtype=torch.complex128, device="cpu").contiguous()
    ).tolist()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/experiments/configs/phase37_n4_d6_k4.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/phase37_slater_source_solver/resumed.pt"),
    )
    parser.add_argument(
        "--clean-checkpoint",
        type=Path,
        default=Path("checkpoints/phase37_slater_source_solver/clean.pt"),
    )
    parser.add_argument(
        "--phase28-artifact",
        type=Path,
        default=Path(
            "docs/experiments/results/phase28_soft_coulomb_transferability.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase37_slater_source_solver.json"
        ),
    )
    args = parser.parse_args()
    if any(path.exists() for path in (args.checkpoint, args.clean_checkpoint, args.output)):
        raise ValueError("registered Phase 37 output/checkpoint already exists")

    config, config_record = load_slater_source_command_config(args.config)
    thresholds = config_record["acceptance"]
    partial = run_slater_source_adaptive_solver(
        config,
        checkpoint_path=args.checkpoint,
        max_adaptive_stages_this_call=1,
    )
    if partial["completed"] or partial["current_terms"] != 2:
        raise RuntimeError("registered Phase 37 interruption did not stop at K2")
    resumed = run_slater_source_adaptive_solver(
        config, checkpoint_path=args.checkpoint, resume=True
    )
    clean = run_slater_source_adaptive_solver(
        config, checkpoint_path=args.clean_checkpoint
    )
    validate_slater_source_result(resumed, require_completed=True)
    validate_slater_source_result(clean, require_completed=True)

    # Truth and historical comparators are opened only after both FEMPS runs.
    dense_truth = _dense_ci(config.basis_order)
    phase28 = json.loads(args.phase28_artifact.read_text(encoding="utf-8"))
    phase28_point = phase28["d6_blind_multiseed"][0]
    resumed_energies = [
        stage["optimizer_result"]["energy"] for stage in resumed["stages"]
    ]
    clean_energies = [
        stage["optimizer_result"]["energy"] for stage in clean["stages"]
    ]
    energy_differences = [
        abs(left - right)
        for left, right in zip(resumed_energies, clean_energies, strict=True)
    ]
    resumed_candidates = [
        stage["selected_candidate"] for stage in resumed["stages"][1:]
    ]
    clean_candidates = [
        stage["selected_candidate"] for stage in clean["stages"][1:]
    ]
    stage_points = [stage["optimizer_result"] for stage in resumed["stages"]]
    final_error = resumed_energies[-1] - dense_truth["energy"]
    source_error = resumed_energies[0] - dense_truth["energy"]
    energy_nonincreasing = all(
        right <= left + thresholds["energy_nesting_tolerance"]
        for left, right in zip(resumed_energies, resumed_energies[1:])
    )
    stage_scientific_records = all(
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
    total_resumed_time = (
        partial["total_elapsed_seconds_this_call"]
        + resumed["total_elapsed_seconds_this_call"]
    )
    source_boundary = all(
        not result["source_construction"]["historical_checkpoint_used"]
        and not result["source_construction"]["ci_initializer_used"]
        for result in (partial, resumed, clean)
    )
    acceptance = {
        "clean_and_resumed_complete_pass": resumed["completed"] and clean["completed"],
        "registered_interruption_pass": (
            not partial["completed"]
            and partial["current_terms"] == 2
            and resumed["resumed"]
            and resumed["stages"][1] == partial["stages"][1]
        ),
        "clean_source_boundary_pass": source_boundary,
        "candidate_schedule_reproduced_pass": resumed_candidates == clean_candidates,
        "clean_resume_energy_match_pass": (
            max(energy_differences) <= thresholds["resume_energy_tolerance"]
        ),
        "energy_nonincreasing_pass": energy_nonincreasing,
        "source_accuracy_pass": (
            abs(source_error) <= thresholds["source_ci_error_maximum"]
            and stage_points[0]["energy_variance"]
            <= thresholds["source_variance_maximum"]
        ),
        "final_accuracy_pass": (
            abs(final_error) <= thresholds["final_ci_error_maximum"]
            and stage_points[-1]["energy_variance"]
            <= thresholds["final_variance_maximum"]
        ),
        "stage_scientific_records_pass": stage_scientific_records,
        "operator_factorization_pass": (
            resumed["operator_metadata"]["dense_relative_factorization_error"]
            <= thresholds["factorization_error_maximum"]
        ),
        "command_resource_pass": (
            total_resumed_time <= thresholds["command_wall_time_maximum_seconds"]
            and clean["total_elapsed_seconds_this_call"]
            <= thresholds["command_wall_time_maximum_seconds"]
            and max(
                partial["peak_cpu_rss_bytes"],
                resumed["peak_cpu_rss_bytes"],
                clean["peak_cpu_rss_bytes"],
            )
            <= thresholds["peak_cpu_rss_maximum_bytes"]
        ),
        "external_cap_boundary_pass": (
            resumed["automatic_stopping_rule"] == "not_admitted"
            and resumed["external_max_terms_required"]
        ),
    }
    acceptance["phase37_slater_source_solver_pass"] = all(acceptance.values())

    source_files = (
        "docs/decisions/0026-preregister-clean-slater-source-solver.md",
        "docs/experiments/configs/phase37_n4_d6_k4.json",
        "src/femps/algorithms/slater_source_contract.py",
        "src/femps/algorithms/slater_source_training.py",
        "src/femps/algorithms/adaptive_diagonal_path_contract.py",
        "src/femps/algorithms/adaptive_diagonal_path_training.py",
        "scripts/run_femps_slater_source_solver.py",
        "scripts/benchmark_phase37_slater_source_solver.py",
        "scripts/verify_phase37_slater_source_solver.py",
    )
    artifact = {
        "schema_version": 1,
        "experiment": "phase37_clean_slater_source_solver",
        "evidence_level": "numerical",
        "scientific_boundary": resumed["scientific_boundary"],
        "registered_config": config_record,
        "source_hash_policy": "UTF-8 text normalized to LF before SHA-256",
        "source_hashes": {
            path: _text_sha256(Path(path)) for path in source_files
        },
        "phase28_artifact": {
            "path": str(args.phase28_artifact),
            "normalized_text_sha256": _text_sha256(args.phase28_artifact),
        },
        "partial_call": {
            "completed": partial["completed"],
            "current_terms": partial["current_terms"],
            "total_elapsed_seconds_this_call": partial[
                "total_elapsed_seconds_this_call"
            ],
            "peak_cpu_rss_bytes": partial["peak_cpu_rss_bytes"],
        },
        "resumed_result": resumed,
        "clean_result": clean,
        "stage_orbitals": _stage_orbitals(args.checkpoint, config.max_terms),
        "dense_ci_comparator": dense_truth,
        "phase28_manual_comparator": {
            "point_id": phase28_point["point_id"],
            "energy": phase28_point["energy"],
            "error_vs_dense_quadrature_ci": phase28_point[
                "error_vs_dense_quadrature_ci"
            ],
            "energy_variance": phase28_point["energy_variance"],
        },
        "comparison": {
            "resumed_candidates": resumed_candidates,
            "clean_candidates": clean_candidates,
            "resumed_energies": resumed_energies,
            "clean_energies": clean_energies,
            "energy_absolute_differences": energy_differences,
            "source_error_vs_dense_ci": source_error,
            "final_error_vs_dense_ci": final_error,
            "final_energy_difference_vs_phase28_manual_K4": (
                resumed_energies[-1] - phase28_point["energy"]
            ),
            "ordinary_particle_tt_ranks": [
                point["ordinary_particle_tt_ranks"] for point in stage_points
            ],
            "femps_correlation_multiplicity": list(
                range(1, config.max_terms + 1)
            ),
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
                "accepted": acceptance["phase37_slater_source_solver_pass"],
                "energies": resumed_energies,
                "selected_candidates": resumed_candidates,
                "final_error_vs_dense_ci": final_error,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
