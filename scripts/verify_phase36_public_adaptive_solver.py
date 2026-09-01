"""Independently verify the Phase 36 public adaptive-solver artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from femps.algorithms import (  # noqa: E402
    select_adaptive_diagonal_path_term,
    solve_generalized_hermitian,
    validate_adaptive_diagonal_path_result,
)
from femps.exterior import (  # noqa: E402
    diagonal_path_exterior_coefficients,
    diagonal_path_hamiltonian_matrices,
)
from femps.hamiltonians import FactorizedTwoBodyOperator  # noqa: E402
from scripts.benchmark_phase34_adaptive_k_growth import (  # noqa: E402
    CONDITION_THRESHOLD,
    CPU_RSS_CAP_BYTES,
    OVERLAP_THRESHOLD,
    POOL_SIZE,
    WALL_TIME_CAP_SECONDS,
    _operators,
    _truth_data,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decode_complex(values: list) -> torch.Tensor:
    real_view = torch.tensor(values, dtype=torch.float64)
    if real_view.shape[-1] != 2:
        raise AssertionError("serialized complex tensor lacks real/imag axis")
    return torch.view_as_complex(real_view.contiguous())


def _hash_tensor(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    _hash_tensor(digest, tensor)
    return digest.hexdigest()


def _operator_sha256(
    one_body: torch.Tensor, interaction: FactorizedTwoBodyOperator | None
) -> str:
    digest = hashlib.sha256()
    _hash_tensor(digest, one_body)
    if interaction is None:
        digest.update(b"no_two_body_operator")
    else:
        _hash_tensor(digest, interaction.left)
        _hash_tensor(digest, interaction.right)
        _hash_tensor(digest, interaction.weights)
    return digest.hexdigest()


def _assert_close(observed: float, expected: float, tolerance: float, label: str) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(
            f"{label}: {observed!r} differs from {expected!r} by more than {tolerance}"
        )


def _source_hashes(artifact: dict) -> None:
    sources = artifact["sources"]
    paths = {
        "phase32_artifact_sha256": Path(
            "docs/experiments/results/phase32_n6_convergence.json"
        ),
        "phase35_artifact_sha256": Path(
            "docs/experiments/results/phase35_adaptive_pool_stability.json"
        ),
        "adaptive_contract_sha256": Path(
            "src/femps/algorithms/adaptive_diagonal_path_contract.py"
        ),
        "adaptive_training_sha256": Path(
            "src/femps/algorithms/adaptive_diagonal_path_training.py"
        ),
        "growth_sha256": Path("src/femps/algorithms/diagonal_path_growth.py"),
        "training_sha256": Path("src/femps/algorithms/diagonal_path_training.py"),
        "runner_sha256": Path(
            "scripts/benchmark_phase36_public_adaptive_solver.py"
        ),
        "adr_sha256": Path(
            "docs/decisions/0025-preregister-public-adaptive-solver.md"
        ),
    }
    for key, path in paths.items():
        if _sha256(path) != sources[key]:
            raise AssertionError(f"Phase 36 source hash mismatch: {key}")
    phase32 = json.loads(paths["phase32_artifact_sha256"].read_text(encoding="utf-8"))
    source = next(
        point
        for point in phase32["points"]
        if point["point_id"] == "N6_D12_K4_seed3212_from_D10"
    )
    if source["checkpoint_sha256"] != sources["phase32_source_checkpoint_sha256"]:
        raise AssertionError("Phase 32 checkpoint lineage hash mismatch")


def _fixed_candidate_minimum(stage: dict) -> None:
    candidates = stage["growth"]["candidates"]
    admitted = [candidate for candidate in candidates if candidate["admitted"]]
    selected = min(
        admitted,
        key=lambda candidate: (
            candidate["predicted_energy"],
            candidate["candidate_index"],
        ),
    )
    if selected["candidate_index"] != stage["selected_candidate"]:
        raise AssertionError("stored selected candidate is not the admitted minimum")
    _assert_close(
        selected["predicted_improvement"],
        stage["predicted_improvement"],
        1e-14,
        "selected predicted improvement",
    )


def verify_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["schema_version"] != 1 or artifact["evidence_level"] != "numerical":
        raise AssertionError("unsupported Phase 36 artifact schema/evidence label")
    _source_hashes(artifact)
    registered = artifact["registered_config"]
    if (
        registered["particles"],
        registered["basis_order"],
        registered["source_terms"],
        registered["max_terms"],
        registered["pool_size"],
    ) != (6, 12, 4, 6, POOL_SIZE):
        raise AssertionError("Phase 36 physical configuration changed")
    if registered["seed_schedule"] != [[5, 3511, 3511], [6, 3512, 3512]]:
        raise AssertionError("Phase 36 seed schedule changed")

    public = artifact["public_result"]
    validate_adaptive_diagonal_path_result(public, require_completed=True)
    if public["current_terms"] != 6 or public["start_terms"] != 4:
        raise AssertionError("public result did not execute K4-to-K6")
    if public["automatic_stopping_rule"] != "not_admitted":
        raise AssertionError("artifact improperly admitted automatic stopping")
    partial = artifact["partial_call"]
    resume_pass = (
        partial["current_terms"] == 5
        and not partial["completed"]
        and partial["stages_completed_this_call"] == 1
        and partial["outer_checkpoint_current_terms"] == 5
        and public["resumed"]
        and public["stages_completed_this_call"] == 1
        and artifact["outer_checkpoint"]
        == {"current_terms": 6, "completed": True, "stages": 2}
    )

    one_body, interaction, diagnostics = _operators()
    if diagnostics != artifact["operator_diagnostics"]:
        raise AssertionError("operator diagnostics changed")
    source = _decode_complex(artifact["source_orbitals"])
    source_identity = public["source_identity"]
    if _tensor_sha256(source) != source_identity["orbitals_sha256"]:
        raise AssertionError("committed source orbitals do not match source identity")
    if _operator_sha256(one_body, interaction) != public["operator_identity"][
        "operator_sha256"
    ]:
        raise AssertionError("reconstructed operator identity mismatch")

    stage_orbitals = {
        record["terms"]: _decode_complex(record["values"])
        for record in artifact["stage_orbitals"]
    }
    phase35 = json.loads(
        Path("docs/experiments/results/phase35_adaptive_pool_stability.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = phase35["lineages"][0]
    dense_hamiltonian, truth_controls = _truth_data(one_body)
    ci_energy = truth_controls["direct_ci"]["energy"]

    current = source
    rebuilt = []
    stage_gates = []
    expected_points = {5: frozen["K5"], 6: frozen["K6"]}
    expected_growth = {5: frozen["growth"]["K4_to_K5"], 6: frozen["growth"]["K5_to_K6"]}
    for stage in public["stages"]:
        target = stage["target_terms"]
        _fixed_candidate_minimum(stage)
        growth = select_adaptive_diagonal_path_term(
            current,
            one_body,
            interaction,
            pool_size=registered["pool_size"],
            seed=stage["candidate_seed"],
            overlap_relative_threshold=OVERLAP_THRESHOLD,
            condition_threshold=CONDITION_THRESHOLD,
        )
        if growth.selected_candidate != stage["selected_candidate"]:
            raise AssertionError("reconstructed candidate selection mismatch")
        if growth.selected_candidate != expected_growth[target]["selected_candidate"]:
            raise AssertionError("public selection differs from frozen Phase 35")
        _assert_close(
            growth.predicted_improvement,
            stage["predicted_improvement"],
            1e-12,
            f"K{target} predicted improvement",
        )
        if float(torch.max(torch.abs(growth.orbitals[: target - 1] - current))) > 1e-13:
            raise AssertionError("adaptive growth lost source nesting")

        orbitals = stage_orbitals[target]
        if _tensor_sha256(orbitals) != stage["optimized_orbitals_sha256"]:
            raise AssertionError("optimized orbital hash mismatch")
        overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
            orbitals,
            one_body,
            two_body_left=interaction.left,
            two_body_right=interaction.right,
            two_body_weights=interaction.weights,
        )
        solved = solve_generalized_hermitian(
            hamiltonian,
            overlap,
            relative_threshold=OVERLAP_THRESHOLD,
        )
        energy = float(solved.energy)
        result = stage["optimizer_result"]
        _assert_close(energy, result["energy"], 1e-11, f"K{target} factorized energy")
        _assert_close(
            energy,
            expected_points[target]["energy"],
            registered["energy_match_tolerance"],
            f"K{target} frozen energy",
        )
        coefficients = diagonal_path_exterior_coefficients(
            orbitals, solved.amplitudes
        )
        norm = torch.vdot(coefficients, coefficients).real
        acted = dense_hamiltonian @ coefficients
        dense_energy_tensor = (torch.vdot(coefficients, acted) / norm).real
        residual = acted - dense_energy_tensor * coefficients
        variance = float(torch.vdot(residual, residual).real / norm)
        dense_energy = float(dense_energy_tensor)
        _assert_close(dense_energy, energy, 1e-10, f"K{target} dense energy")
        _assert_close(
            variance,
            expected_points[target]["dense_quadrature_energy_variance"],
            1e-10,
            f"K{target} variance",
        )
        counts = result["structural_counts"]
        stage_pass = (
            result["completed"]
            and result["structural_antisymmetry_residual"] == 0.0
            and counts["enumerated_virtual_paths"] == 0
            and counts["materialized_particle_coefficients"] == 0
            and result["norm_error"] <= registered["norm_tolerance"]
            and result["retained_condition_number"] <= registered["condition_cap"]
            and result["total_elapsed_seconds_this_call"]
            <= registered["wall_time_cap_seconds"]
            and result["peak_cpu_rss_bytes"] <= registered["cpu_rss_cap_bytes"]
        )
        stage_gates.append(stage_pass)
        rebuilt.append(
            {
                "terms": target,
                "selected_candidate": growth.selected_candidate,
                "energy": energy,
                "dense_energy": dense_energy,
                "error_vs_CI": dense_energy - ci_energy,
                "variance": variance,
                "norm_error": float(abs(norm - 1.0)),
            }
        )
        current = orbitals

    comparison = artifact["comparison"]
    energy_match = max(comparison["energy_absolute_differences"]) <= registered[
        "energy_match_tolerance"
    ]
    selection_match = all(comparison["selected_candidate_matches"])
    recomputed_acceptance = {
        "public_contract_pass": public["completed"] and public["current_terms"] == 6,
        "stage_resume_pass": resume_pass,
        "phase35_selection_match_pass": selection_match,
        "phase35_energy_match_pass": energy_match,
        "stage_scientific_records_pass": all(stage_gates),
        "external_cap_boundary_pass": (
            public["automatic_stopping_rule"] == "not_admitted"
            and public["external_max_terms_required"]
        ),
    }
    recomputed_acceptance["phase36_public_adaptive_solver_pass"] = all(
        recomputed_acceptance.values()
    )
    if recomputed_acceptance != artifact["acceptance"]:
        raise AssertionError("Phase 36 acceptance record does not recompute")
    return {
        "verified": recomputed_acceptance["phase36_public_adaptive_solver_pass"],
        "rebuilt_stages": rebuilt,
        "maximum_energy_difference_vs_phase35": max(
            comparison["energy_absolute_differences"]
        ),
        "final_error_vs_CI": rebuilt[-1]["error_vs_CI"],
        "final_variance": rebuilt[-1]["variance"],
        "automatic_stopping_rule": public["automatic_stopping_rule"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path(
            "docs/experiments/results/phase36_public_adaptive_solver.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
