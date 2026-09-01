"""Run the preregistered Phase 36 public adaptive-solver reproduction."""

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
    AdaptiveDiagonalPathConfig,
    AdaptiveDiagonalPathStageConfig,
    canonical_slater_orbitals,
    load_adaptive_diagonal_path_checkpoint,
    load_diagonal_path_checkpoint,
    run_bounded_adaptive_diagonal_path,
    validate_adaptive_diagonal_path_result,
)

from scripts.benchmark_phase34_adaptive_k_growth import (
    CONDITION_THRESHOLD,
    CPU_RSS_CAP_BYTES,
    OVERLAP_THRESHOLD,
    POOL_SIZE,
    WALL_TIME_CAP_SECONDS,
    _config,
    _operators,
)


SEED_SCHEDULE = ((5, 3511, 3511), (6, 3512, 3512))
ENERGY_MATCH_TOLERANCE = 1e-11
NORM_TOLERANCE = 1e-10


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complex_tensor(tensor: torch.Tensor) -> list:
    return torch.view_as_real(
        tensor.detach().to(dtype=torch.complex128, device="cpu").contiguous()
    ).tolist()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _optimizer_checkpoint(outer: Path, terms: int) -> Path:
    return outer.with_name(f"{outer.stem}.K{terms}.optimizer.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase32-artifact",
        type=Path,
        default=Path("docs/experiments/results/phase32_n6_convergence.json"),
    )
    parser.add_argument(
        "--phase35-artifact",
        type=Path,
        default=Path(
            "docs/experiments/results/phase35_adaptive_pool_stability.json"
        ),
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/phase32_n6_convergence/N6_D12_K4_seed3212_from_D10.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/phase36_public_adaptive_solver/lineage1_canonical_once.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase36_public_adaptive_solver.json"
        ),
    )
    args = parser.parse_args()

    phase32 = json.loads(args.phase32_artifact.read_text(encoding="utf-8"))
    source_record = next(
        point
        for point in phase32["points"]
        if point["point_id"] == "N6_D12_K4_seed3212_from_D10"
    )
    if not phase32["acceptance"]["phase32_convergence_pass"]:
        raise RuntimeError("Phase 32 source gate is not accepted")
    if _sha256(args.source_checkpoint) != source_record["checkpoint_sha256"]:
        raise RuntimeError("Phase 32 source checkpoint hash mismatch")
    if args.checkpoint.exists():
        raise RuntimeError("Phase 36 outer checkpoint already exists")

    one_body, interaction, operator_diagnostics = _operators()
    source_raw = load_diagonal_path_checkpoint(args.source_checkpoint)["best_raw"]
    source = canonical_slater_orbitals(source_raw)
    schedule = AdaptiveDiagonalPathConfig(
        max_terms=6,
        pool_size=POOL_SIZE,
        stages=tuple(
            AdaptiveDiagonalPathStageConfig(target, candidate, optimizer)
            for target, candidate, optimizer in SEED_SCHEDULE
        ),
        overlap_relative_threshold=OVERLAP_THRESHOLD,
        condition_threshold=CONDITION_THRESHOLD,
    )
    template = _config(4, 0)

    partial = run_bounded_adaptive_diagonal_path(
        source_raw,
        one_body,
        interaction,
        template,
        schedule,
        source_id="phase32_N6_D12_K4_seed3212_from_D10",
        operator_id="soft_N6_D12_Q128_physical_svd_phase36",
        checkpoint_path=args.checkpoint,
        max_stages_this_call=1,
    )
    partial_checkpoint = load_adaptive_diagonal_path_checkpoint(args.checkpoint)
    if partial["completed"] or partial_checkpoint["current_terms"] != 5:
        raise RuntimeError("registered interruption did not stop after K5")

    final = run_bounded_adaptive_diagonal_path(
        source_raw,
        one_body,
        interaction,
        template,
        schedule,
        source_id="phase32_N6_D12_K4_seed3212_from_D10",
        operator_id="soft_N6_D12_Q128_physical_svd_phase36",
        checkpoint_path=args.checkpoint,
        resume=True,
    )
    validate_adaptive_diagonal_path_result(final, require_completed=True)
    final_checkpoint = load_adaptive_diagonal_path_checkpoint(args.checkpoint)

    # Only now may the frozen Phase 35 result be read.
    phase35 = json.loads(args.phase35_artifact.read_text(encoding="utf-8"))
    frozen = phase35["lineages"][0]
    phase35_points = [frozen["K5"], frozen["K6"]]
    energy_differences = [
        abs(stage["optimizer_result"]["energy"] - point["energy"])
        for stage, point in zip(final["stages"], phase35_points, strict=True)
    ]
    selected_match = [
        stage["selected_candidate"]
        == frozen["growth"][f"K{stage['target_terms'] - 1}_to_K{stage['target_terms']}"][
            "selected_candidate"
        ]
        for stage in final["stages"]
    ]
    stage_orbitals = []
    for target, _, _ in SEED_SCHEDULE:
        optimizer_payload = load_diagonal_path_checkpoint(
            _optimizer_checkpoint(args.checkpoint, target)
        )
        stage_orbitals.append(
            canonical_slater_orbitals(optimizer_payload["best_raw"])
        )

    stage_gates = []
    for stage in final["stages"]:
        result = stage["optimizer_result"]
        counts = result["structural_counts"]
        stage_gates.append(
            result["completed"]
            and result["structural_antisymmetry_residual"] == 0.0
            and counts["enumerated_virtual_paths"] == 0
            and counts["materialized_particle_coefficients"] == 0
            and result["norm_error"] <= NORM_TOLERANCE
            and result["retained_condition_number"] <= CONDITION_THRESHOLD
            and result["total_elapsed_seconds_this_call"] <= WALL_TIME_CAP_SECONDS
            and result["peak_cpu_rss_bytes"] <= CPU_RSS_CAP_BYTES
        )
    acceptance = {
        "public_contract_pass": final["completed"] and final["current_terms"] == 6,
        "stage_resume_pass": (
            not partial["completed"]
            and partial["current_terms"] == 5
            and final["resumed"]
            and final["stages_completed_this_call"] == 1
            and final["stages"][0] == partial["stages"][0]
        ),
        "phase35_selection_match_pass": all(selected_match),
        "phase35_energy_match_pass": max(energy_differences) <= ENERGY_MATCH_TOLERANCE,
        "stage_scientific_records_pass": all(stage_gates),
        "external_cap_boundary_pass": (
            final["automatic_stopping_rule"] == "not_admitted"
            and final["external_max_terms_required"]
        ),
    }
    acceptance["phase36_public_adaptive_solver_pass"] = all(acceptance.values())

    sources = {
        "phase32_artifact_sha256": _sha256(args.phase32_artifact),
        "phase35_artifact_sha256": _sha256(args.phase35_artifact),
        "phase32_source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "adaptive_contract_sha256": _sha256(
            Path("src/femps/algorithms/adaptive_diagonal_path_contract.py")
        ),
        "adaptive_training_sha256": _sha256(
            Path("src/femps/algorithms/adaptive_diagonal_path_training.py")
        ),
        "growth_sha256": _sha256(
            Path("src/femps/algorithms/diagonal_path_growth.py")
        ),
        "training_sha256": _sha256(
            Path("src/femps/algorithms/diagonal_path_training.py")
        ),
        "runner_sha256": _sha256(Path(__file__)),
        "adr_sha256": _sha256(
            Path("docs/decisions/0025-preregister-public-adaptive-solver.md")
        ),
    }
    artifact = {
        "schema_version": 1,
        "experiment": "phase36_public_bounded_adaptive_solver",
        "evidence_level": "numerical",
        "scientific_boundary": final["scientific_boundary"],
        "registered_config": {
            "particles": 6,
            "basis_order": 12,
            "source_terms": 4,
            "max_terms": 6,
            "pool_size": POOL_SIZE,
            "seed_schedule": [list(item) for item in SEED_SCHEDULE],
            "optimizer_template": final["optimizer_template"],
            "energy_match_tolerance": ENERGY_MATCH_TOLERANCE,
            "norm_tolerance": NORM_TOLERANCE,
            "condition_cap": CONDITION_THRESHOLD,
            "wall_time_cap_seconds": WALL_TIME_CAP_SECONDS,
            "cpu_rss_cap_bytes": CPU_RSS_CAP_BYTES,
        },
        "sources": sources,
        "operator_diagnostics": operator_diagnostics,
        "source_orbitals": _complex_tensor(source),
        "stage_orbitals": [
            {"terms": target, "values": _complex_tensor(orbitals)}
            for (target, _, _), orbitals in zip(
                SEED_SCHEDULE, stage_orbitals, strict=True
            )
        ],
        "partial_call": {
            "current_terms": partial["current_terms"],
            "completed": partial["completed"],
            "stages_completed_this_call": partial["stages_completed_this_call"],
            "outer_checkpoint_current_terms": partial_checkpoint["current_terms"],
        },
        "public_result": final,
        "outer_checkpoint": {
            "current_terms": final_checkpoint["current_terms"],
            "completed": final_checkpoint["completed"],
            "stages": len(final_checkpoint["stages"]),
        },
        "frozen_phase35_lineage1": {
            "selected_candidates": [
                frozen["growth"]["K4_to_K5"]["selected_candidate"],
                frozen["growth"]["K5_to_K6"]["selected_candidate"],
            ],
            "energies": [point["energy"] for point in phase35_points],
        },
        "comparison": {
            "selected_candidate_matches": selected_match,
            "energy_absolute_differences": energy_differences,
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
                "accepted": acceptance["phase36_public_adaptive_solver_pass"],
                "energies": [
                    stage["optimizer_result"]["energy"] for stage in final["stages"]
                ],
                "selected_candidates": [
                    stage["selected_candidate"] for stage in final["stages"]
                ],
                "energy_differences": energy_differences,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
