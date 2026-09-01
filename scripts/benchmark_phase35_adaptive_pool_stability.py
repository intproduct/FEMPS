"""Run the preregistered Phase 35 adaptive-pool stability experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

from femps.algorithms import (
    assess_term_pruning,
    canonical_slater_orbitals,
    load_diagonal_path_checkpoint,
    run_diagonal_path_variable_projection,
    select_adaptive_diagonal_path_term,
    solve_generalized_hermitian,
)
from femps.exterior import diagonal_path_hamiltonian_matrices

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.benchmark_phase34_adaptive_k_growth import (
    CONDITION_THRESHOLD,
    CPU_RSS_CAP_BYTES,
    DIMENSION,
    ENERGY_MONOTONICITY_TOLERANCE,
    OVERLAP_THRESHOLD,
    PARTICLES,
    POOL_SIZE,
    PRUNING_ENERGY_TOLERANCE,
    WALL_TIME_CAP_SECONDS,
    _audit_state,
    _config,
    _growth_record,
    _operators,
    _truth_data,
)


SEED_PAIRS = ((3511, 3512), (3521, 3522), (3531, 3532))
STOP_PREDICTED_THRESHOLD = 1e-8
STOP_REALIZED_THRESHOLD = 1e-8
K6_CI_ERROR_CAP = 1.1e-4
K6_VARIANCE_CAP = 1.5e-3
K6_ENERGY_SPREAD_CAP = 1e-4
FACTORIZED_DENSE_TOLERANCE = 1e-10
OPERATOR_FACTORIZATION_TOLERANCE = 1e-11
QUADRATURE_TOLERANCE = 2e-12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_point(
    point_id: str,
    terms: int,
    seed: int,
    initial_orbitals: torch.Tensor,
    checkpoint_dir: Path,
    one_body: torch.Tensor,
    interaction: object,
    lineage: dict,
) -> tuple[dict, torch.Tensor]:
    checkpoint = checkpoint_dir / f"{point_id}.pt"
    result = run_diagonal_path_variable_projection(
        _config(terms, seed),
        checkpoint_path=checkpoint,
        initial_orbitals=initial_orbitals,
        operators=(one_body, interaction),
        operator_id="soft_N6_D12_Q128_physical_svd_phase35",
    )
    payload = load_diagonal_path_checkpoint(checkpoint)
    orbitals = canonical_slater_orbitals(payload["best_raw"])
    result["point_id"] = point_id
    result["initialization_lineage"] = lineage
    result["checkpoint_sha256"] = _sha256(checkpoint)
    print(point_id, result["energy"], flush=True)
    return result, orbitals


def _compact_point(point: dict) -> dict:
    keys = (
        "schema_version",
        "method",
        "evidence_level",
        "config",
        "environment",
        "operator",
        "operator_id",
        "initialization",
        "resumed",
        "completed",
        "completed_steps",
        "point_id",
        "initialization_lineage",
        "checkpoint_sha256",
        "energy",
        "dense_quadrature_energy",
        "dense_quadrature_energy_variance",
        "dense_quadrature_norm_error",
        "error_vs_direct_ci",
        "factorized_vs_dense_energy_difference",
        "norm",
        "norm_error",
        "structural_antisymmetry_residual",
        "materialized_antisymmetry_residual",
        "retained_rank",
        "discarded_rank",
        "retained_condition_number",
        "raw_overlap_condition_number",
        "generalized_residual_norm",
        "structural_counts",
        "transition_diagnostics",
        "refinement",
        "total_elapsed_seconds_this_call",
        "peak_cpu_rss_bytes",
        "peak_cpu_rss_delta_bytes",
        "peak_cuda_memory_bytes",
        "cpu_memory",
        "ordinary_particle_tt_ranks_compact",
        "ordinary_particle_tt_storage_scalars",
        "femps_stored_parameter_scalars",
        "raw_exterior_coefficients",
    )
    return {key: point[key] for key in keys}


def _historical_cold_summary(point: dict) -> dict:
    return {
        "point_id": point["point_id"],
        "energy": point["dense_quadrature_energy"],
        "error_vs_direct_ci": point["error_vs_direct_ci"],
        "energy_variance": point["dense_quadrature_energy_variance"],
        "condition": point["retained_condition_number"],
        "time_seconds": point["total_elapsed_seconds_this_call"],
        "peak_cpu_rss_bytes": point["peak_cpu_rss_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase32-artifact",
        type=Path,
        default=Path("docs/experiments/results/phase32_n6_convergence.json"),
    )
    parser.add_argument(
        "--phase34-artifact",
        type=Path,
        default=Path("docs/experiments/results/phase34_adaptive_k_growth.json"),
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/phase32_n6_convergence/N6_D12_K4_seed3212_from_D10.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase35_adaptive_pool_stability"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase35_adaptive_pool_stability.json"
        ),
    )
    args = parser.parse_args()

    phase32 = json.loads(args.phase32_artifact.read_text(encoding="utf-8"))
    if not phase32["acceptance"]["phase32_convergence_pass"]:
        raise RuntimeError("Phase 32 source convergence gate did not pass")
    source_record = next(
        point
        for point in phase32["points"]
        if point["point_id"] == "N6_D12_K4_seed3212_from_D10"
    )
    if _sha256(args.source_checkpoint) != source_record["checkpoint_sha256"]:
        raise RuntimeError("Phase 32 source checkpoint hash mismatch")

    one_body, interaction, operator_diagnostics = _operators()
    source_payload = load_diagonal_path_checkpoint(args.source_checkpoint)
    source_orbitals = canonical_slater_orbitals(source_payload["best_raw"])
    source_overlap, source_hamiltonian = diagonal_path_hamiltonian_matrices(
        source_orbitals,
        one_body,
        two_body_left=interaction.left,
        two_body_right=interaction.right,
        two_body_weights=interaction.weights,
    )
    source_solved = solve_generalized_hermitian(
        source_hamiltonian, source_overlap, relative_threshold=OVERLAP_THRESHOLD
    )
    if abs(float(source_solved.energy) - source_record["energy"]) > 1e-10:
        raise RuntimeError("Phase 32 source state does not reproduce its energy")

    # All six optimizations are completed before _truth_data is called or the
    # Phase 34 artifact is opened.
    staged_lineages = []
    for lineage_index, (seed_k5, seed_k6) in enumerate(SEED_PAIRS, start=1):
        growth_k5 = select_adaptive_diagonal_path_term(
            source_orbitals,
            one_body,
            interaction,
            pool_size=POOL_SIZE,
            seed=seed_k5,
            overlap_relative_threshold=OVERLAP_THRESHOLD,
            condition_threshold=CONDITION_THRESHOLD,
        )
        k5, k5_orbitals = _run_point(
            f"N6_D12_K5_seed{seed_k5}_lineage{lineage_index}",
            5,
            seed_k5,
            growth_k5.orbitals,
            args.checkpoint_dir,
            one_body,
            interaction,
            {
                "kind": "truth_free_adaptive_K4_plus_one",
                "lineage": lineage_index,
                "seed": seed_k5,
                "selected_candidate": growth_k5.selected_candidate,
                "truth_state_used": False,
            },
        )
        growth_k6 = select_adaptive_diagonal_path_term(
            k5_orbitals,
            one_body,
            interaction,
            pool_size=POOL_SIZE,
            seed=seed_k6,
            overlap_relative_threshold=OVERLAP_THRESHOLD,
            condition_threshold=CONDITION_THRESHOLD,
        )
        k6, k6_orbitals = _run_point(
            f"N6_D12_K6_seed{seed_k6}_lineage{lineage_index}",
            6,
            seed_k6,
            growth_k6.orbitals,
            args.checkpoint_dir,
            one_body,
            interaction,
            {
                "kind": "truth_free_adaptive_K5_plus_one",
                "lineage": lineage_index,
                "seed": seed_k6,
                "selected_candidate": growth_k6.selected_candidate,
                "truth_state_used": False,
            },
        )
        k6_overlap, k6_hamiltonian = diagonal_path_hamiltonian_matrices(
            k6_orbitals,
            one_body,
            two_body_left=interaction.left,
            two_body_right=interaction.right,
            two_body_weights=interaction.weights,
        )
        pruning = assess_term_pruning(
            k6_hamiltonian,
            k6_overlap,
            condition_threshold=CONDITION_THRESHOLD,
            energy_tolerance=PRUNING_ENERGY_TOLERANCE,
            relative_threshold=OVERLAP_THRESHOLD,
        )
        staged_lineages.append(
            {
                "lineage_id": lineage_index,
                "seeds": {"K5": seed_k5, "K6": seed_k6},
                "growth_k5": growth_k5,
                "growth_k6": growth_k6,
                "k5": k5,
                "k5_orbitals": k5_orbitals,
                "k6": k6,
                "k6_orbitals": k6_orbitals,
                "pruning": pruning,
            }
        )

    dense_hamiltonian, controls = _truth_data(one_body)
    phase34 = json.loads(args.phase34_artifact.read_text(encoding="utf-8"))
    if not phase34["acceptance"]["phase34_adaptive_growth_pass"]:
        raise RuntimeError("historical Phase 34 control is not accepted")
    ci_energy = controls["direct_ci"]["energy"]
    audited_source = _audit_state(
        dict(source_record),
        source_orbitals,
        one_body,
        interaction,
        dense_hamiltonian,
        ci_energy,
    )

    lineages = []
    stop_decisions = []
    for staged in staged_lineages:
        k5 = _audit_state(
            staged["k5"],
            staged["k5_orbitals"],
            one_body,
            interaction,
            dense_hamiltonian,
            ci_energy,
        )
        k6 = _audit_state(
            staged["k6"],
            staged["k6_orbitals"],
            one_body,
            interaction,
            dense_hamiltonian,
            ci_energy,
        )
        growth_k5 = _growth_record(staged["growth_k5"])
        growth_k6 = _growth_record(staged["growth_k6"])
        realized_k5 = (
            audited_source["dense_quadrature_energy"]
            - k5["dense_quadrature_energy"]
        )
        realized_k6 = k5["dense_quadrature_energy"] - k6["dense_quadrature_energy"]
        for transition, growth, realized in (
            ("K4_to_K5", growth_k5, realized_k5),
            ("K5_to_K6", growth_k6, realized_k6),
        ):
            predicted_continue = (
                growth["predicted_improvement"] >= STOP_PREDICTED_THRESHOLD
            )
            realized_continue = realized >= STOP_REALIZED_THRESHOLD
            stop_decisions.append(
                {
                    "lineage_id": staged["lineage_id"],
                    "transition": transition,
                    "predicted_improvement": growth["predicted_improvement"],
                    "realized_improvement": realized,
                    "predicted_decision": (
                        "continue" if predicted_continue else "stop"
                    ),
                    "realized_decision": (
                        "continue" if realized_continue else "stop"
                    ),
                    "agreement": predicted_continue == realized_continue,
                }
            )
        lineages.append(
            {
                "lineage_id": staged["lineage_id"],
                "seeds": staged["seeds"],
                "growth": {
                    "K4_to_K5": growth_k5,
                    "K5_to_K6": growth_k6,
                },
                "K5": _compact_point(k5),
                "K6": _compact_point(k6),
                "pruning_assessment": {
                    "should_prune": staged["pruning"].should_prune,
                    "candidate": staged["pruning"].candidate,
                    "energy_penalty": staged["pruning"].energy_penalty,
                    "balanced_condition_number": (
                        staged["pruning"].balanced_condition_number
                    ),
                    "discarded_rank": staged["pruning"].discarded_rank,
                    "reason": staged["pruning"].reason,
                },
                "comparison": {
                    "K5_improvement_over_K4": realized_k5,
                    "K6_improvement_over_K5": realized_k6,
                    "K6_total_improvement_over_K4": (
                        audited_source["dense_quadrature_energy"]
                        - k6["dense_quadrature_energy"]
                    ),
                },
            }
        )

    optimized_points = [
        lineage[point] for lineage in lineages for point in ("K5", "K6")
    ]
    k6_points = [lineage["K6"] for lineage in lineages]
    monotonic_pass = all(
        lineage["K5"]["dense_quadrature_energy"]
        <= audited_source["dense_quadrature_energy"]
        + ENERGY_MONOTONICITY_TOLERANCE
        and lineage["K6"]["dense_quadrature_energy"]
        <= lineage["K5"]["dense_quadrature_energy"]
        + ENERGY_MONOTONICITY_TOLERANCE
        and lineage["comparison"]["K6_total_improvement_over_K4"] >= 1e-8
        for lineage in lineages
    )
    k6_quality_pass = all(
        point["error_vs_direct_ci"] <= K6_CI_ERROR_CAP
        and point["dense_quadrature_energy_variance"] <= K6_VARIANCE_CAP
        for point in k6_points
    )
    k6_energies = [point["dense_quadrature_energy"] for point in k6_points]
    k6_energy_spread = max(k6_energies) - min(k6_energies)
    spread_pass = k6_energy_spread <= K6_ENERGY_SPREAD_CAP
    structural_pass = all(
        point["completed"]
        and point["norm_error"] <= 1e-10
        and point["dense_quadrature_norm_error"] <= 1e-10
        and point["structural_antisymmetry_residual"] <= 1e-12
        and point["retained_rank"] == point["config"]["terms"]
        and point["retained_condition_number"] <= CONDITION_THRESHOLD
        and point["structural_counts"]["enumerated_virtual_paths"] == 0
        and point["structural_counts"]["materialized_particle_coefficients"] == 0
        for point in optimized_points
    )
    truth_pass = all(
        abs(point["factorized_vs_dense_energy_difference"])
        <= FACTORIZED_DENSE_TOLERANCE
        for point in optimized_points
    )
    operator_pass = (
        operator_diagnostics["physical_operator_svd_relative_error"]
        <= OPERATOR_FACTORIZATION_TOLERANCE
        and controls["quadrature_relative_change_Q128_vs_Q160"]
        <= QUADRATURE_TOLERANCE
    )
    resource_pass = all(
        point["total_elapsed_seconds_this_call"] <= WALL_TIME_CAP_SECONDS
        and point["peak_cpu_rss_bytes"] <= CPU_RSS_CAP_BYTES
        for point in optimized_points
    )
    pruning_pass = all(
        not lineage["pruning_assessment"]["should_prune"] for lineage in lineages
    )
    stop_signal_consistent = all(item["agreement"] for item in stop_decisions)
    stop_events = sum(
        item["predicted_decision"] == "stop" for item in stop_decisions
    )
    automatic_stopping_rule_admitted = stop_signal_consistent and stop_events > 0
    accepted = bool(
        monotonic_pass
        and k6_quality_pass
        and spread_pass
        and structural_pass
        and truth_pass
        and operator_pass
        and resource_pass
        and pruning_pass
    )

    artifact = {
        "schema_version": 1,
        "experiment": "phase35_N6_D12_adaptive_pool_stability",
        "evidence_level": "numerical",
        "scientific_boundary": (
            "three fresh 32-candidate K4-to-K6 lineages at one N6,D12 model; "
            "pool stability evidence, not N/D/asymptotic scaling"
        ),
        "registered_config": {
            "N": PARTICLES,
            "D": DIMENSION,
            "K_axis": [4, 5, 6],
            "pool_size": POOL_SIZE,
            "seed_pairs": [list(pair) for pair in SEED_PAIRS],
            "steps": 160,
            "lbfgs_steps": 80,
            "device": "cpu",
            "truth_state_initialization": False,
            "truth_constructed_after_all_optimizations": True,
        },
        "source_records": {
            "phase32_artifact": {
                "path": args.phase32_artifact.as_posix(),
                "sha256": _sha256(args.phase32_artifact),
            },
            "phase32_K4_checkpoint_sha256": _sha256(args.source_checkpoint),
            "phase34_historical_control": {
                "path": args.phase34_artifact.as_posix(),
                "sha256": _sha256(args.phase34_artifact),
            },
            "source_hashes": {
                "growth": _sha256(
                    Path("src/femps/algorithms/diagonal_path_growth.py")
                ),
                "training": _sha256(
                    Path("src/femps/algorithms/diagonal_path_training.py")
                ),
                "phase34_utilities": _sha256(
                    Path("scripts/benchmark_phase34_adaptive_k_growth.py")
                ),
                "benchmark": _sha256(Path(__file__)),
            },
        },
        "operator": {**operator_diagnostics, **controls},
        "source_K4": _compact_point(audited_source),
        "lineages": lineages,
        "historical_phase34_cold_K6": _historical_cold_summary(
            phase34["cold_K6_control"]
        ),
        "stopping_calibration": {
            "rule": "continue iff fixed-span predicted improvement >= 1e-8",
            "predicted_threshold": STOP_PREDICTED_THRESHOLD,
            "realized_threshold": STOP_REALIZED_THRESHOLD,
            "decisions": stop_decisions,
            "all_decisions_consistent": stop_signal_consistent,
            "observed_stop_events": stop_events,
            "automatic_stopping_rule_admitted": automatic_stopping_rule_admitted,
            "conclusion": (
                "automatic stopping admitted"
                if automatic_stopping_rule_admitted
                else "K remains externally capped"
            ),
        },
        "comparison": {
            "K6_energies": k6_energies,
            "K6_energy_spread": k6_energy_spread,
            "maximum_K6_error_vs_CI": max(
                point["error_vs_direct_ci"] for point in k6_points
            ),
            "maximum_K6_variance": max(
                point["dense_quadrature_energy_variance"] for point in k6_points
            ),
            "minimum_K4_to_K6_improvement": min(
                lineage["comparison"]["K6_total_improvement_over_K4"]
                for lineage in lineages
            ),
        },
        "thresholds": {
            "energy_monotonicity_tolerance": ENERGY_MONOTONICITY_TOLERANCE,
            "minimum_K4_to_K6_improvement": 1e-8,
            "K6_CI_error": K6_CI_ERROR_CAP,
            "K6_variance": K6_VARIANCE_CAP,
            "K6_energy_spread": K6_ENERGY_SPREAD_CAP,
            "norm_error": 1e-10,
            "antisymmetry_residual": 1e-12,
            "retained_condition_number": CONDITION_THRESHOLD,
            "factorized_vs_dense_energy": FACTORIZED_DENSE_TOLERANCE,
            "operator_factorization_error": OPERATOR_FACTORIZATION_TOLERANCE,
            "quadrature_relative_change": QUADRATURE_TOLERANCE,
            "peak_cpu_rss_bytes": CPU_RSS_CAP_BYTES,
            "wall_time_seconds_per_point": WALL_TIME_CAP_SECONDS,
        },
        "acceptance": {
            "K_axes_pass": monotonic_pass,
            "K6_quality_pass": k6_quality_pass,
            "K6_spread_pass": spread_pass,
            "structural_pass": structural_pass,
            "truth_reconstruction_pass": truth_pass,
            "operator_pass": operator_pass,
            "resource_pass": resource_pass,
            "pruning_not_triggered_pass": pruning_pass,
            "phase35_pool_stability_pass": accepted,
        },
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
                "output": str(args.output),
                "accepted": accepted,
                "K6_energies": k6_energies,
                "K6_spread": k6_energy_spread,
                "stop_rule_admitted": automatic_stopping_rule_admitted,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
