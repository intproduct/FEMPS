"""Independently verify the restored Phase 39 N4,D8 internal artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from femps.algorithms import (  # noqa: E402
    load_slater_source_command_config,
)
from femps.hamiltonians import (  # noqa: E402
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)
from scripts.benchmark_phase38_clean_source_seed_robustness import (  # noqa: E402
    _canonical_json_sha256,
    _optimizer_failure_count,
)
from scripts.verify_phase37_slater_source_solver import (  # noqa: E402
    verify_artifact as verify_phase37_artifact,
)
from scripts.verify_phase38_clean_source_seed_robustness import (  # noqa: E402
    _close,
    _rebuild_lineage,
    _text_sha256,
)


def verify_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1:
        raise AssertionError("unsupported restored Phase 39 artifact schema")
    if artifact.get("experiment") != "phase39_n4_d8_clean_source_internal":
        raise AssertionError("unexpected restored Phase 39 experiment")
    if artifact.get("evidence_level") != "internal numerical evidence":
        raise AssertionError("D8 result must remain internal numerical evidence")
    if "not a new ansatz" not in artifact.get("scientific_boundary", ""):
        raise AssertionError("D8 scientific boundary is missing")
    for source, expected in artifact["source_hashes"].items():
        if _text_sha256(Path(source)) != expected:
            raise AssertionError(f"normalized source hash mismatch: {source}")

    config_path = Path("docs/experiments/configs/phase39_n4_d8_k4.json")
    config, record = load_slater_source_command_config(config_path)
    if record != artifact["registered_config"]:
        raise AssertionError("registered D8 configuration changed")
    schedule = (
        config.source_optimizer.seed,
        [(stage.candidate_seed, stage.optimizer_seed) for stage in config.stages],
    )
    if schedule != (4001, [(4011, 4012), (4021, 4022), (4031, 4032)]):
        raise AssertionError("registered D8 seed schedule changed")
    if (config.particles, config.basis_order, config.max_terms) != (4, 8, 4):
        raise AssertionError("registered D8 system axes changed")
    thresholds = record["acceptance"]

    historical = artifact["historical_d6_control"]
    historical_path = Path(historical["path"])
    if _text_sha256(historical_path) != historical["normalized_text_sha256"]:
        raise AssertionError("historical D6 control hash changed")
    historical_verified = verify_phase37_artifact(historical_path)
    _close(
        historical_verified["rebuilt_stages"][-1]["energy"],
        historical["final_energy"],
        "historical D6 final energy",
        2e-10,
    )

    one_body = harmonic_pair_hamiltonian(
        config.basis_order,
        kappa=0.0,
        omega=config.omega,
        dtype=torch.complex128,
        device="cpu",
    )[0]
    interaction, diagnostics = soft_coulomb_operator(
        config.basis_order,
        quadrature_order=config.quadrature_order,
        coupling=config.coupling,
        softening=config.softening,
        relative_threshold=config.relative_factor_threshold,
        factorization_backend=config.factorization_backend,
        dtype=torch.complex128,
        device="cpu",
    )
    dense_pair = soft_coulomb_dense_quadrature(
        config.basis_order,
        quadrature_order=config.quadrature_order,
        coupling=config.coupling,
        softening=config.softening,
        dtype=torch.complex128,
        device="cpu",
    )
    dense_hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, config.particles, dense_pair
    )
    dense_ci_energy = float(torch.linalg.eigvalsh(dense_hamiltonian)[0].real)
    _close(
        dense_ci_energy,
        artifact["dense_ci_comparator"]["energy"],
        "D8 dense CI",
        1e-12,
    )
    if diagnostics.dense_relative_factorization_error > thresholds[
        "factorization_error_maximum"
    ]:
        raise AssertionError("D8 operator factorization gate failed")

    results = artifact["results"]
    rebuilt = {
        name: _rebuild_lineage(
            name,
            results[name],
            artifact["stage_orbitals"][name],
            config,
            one_body,
            interaction,
            dense_hamiltonian,
            dense_ci_energy,
        )
        for name in ("resumed", "clean")
    }
    rebuilt_energies = {
        name: [point["energy"] for point in points]
        for name, points in rebuilt.items()
    }
    comparison = artifact["comparison"]
    for name, energies in rebuilt_energies.items():
        for observed, expected in zip(
            energies, comparison["energies"][name], strict=True
        ):
            _close(observed, expected, f"{name} serialized energy", 2e-10)
    differences = [
        abs(left - right)
        for left, right in zip(
            comparison["energies"]["resumed"],
            comparison["energies"]["clean"],
            strict=True,
        )
    ]
    for observed, expected in zip(
        differences,
        comparison["clean_resume_energy_absolute_differences"],
        strict=True,
    ):
        _close(observed, expected, "clean/resume difference", 1e-15)

    execution = artifact["execution"]
    partial = execution["partial"]
    if partial["completed"] or partial["current_terms"] != 2:
        raise AssertionError("registered K2 interruption changed")
    if partial["stage_sha256"] != _canonical_json_sha256(
        results["resumed"]["stages"][:2]
    ):
        raise AssertionError("partial K2 lineage does not match resume")
    if len(execution["public_commands"]) != 3:
        raise AssertionError("unexpected number of public command calls")
    if execution["outcome_dependent_retries"] != 0:
        raise AssertionError("outcome-dependent retry recorded")
    if execution["additional_small_points_authorized"] != 0:
        raise AssertionError("additional small points were authorized")
    if not execution["truth_opened_after_both_d8_optimizations"]:
        raise AssertionError("truth timing boundary failed")

    source_errors = [points[0]["error_vs_CI"] for points in rebuilt.values()]
    final_errors = [points[-1]["error_vs_CI"] for points in rebuilt.values()]
    source_variances = [points[0]["variance"] for points in rebuilt.values()]
    final_variances = [points[-1]["variance"] for points in rebuilt.values()]
    stage_points = [
        stage["optimizer_result"]
        for result in results.values()
        for stage in result["stages"]
    ]
    failures = _optimizer_failure_count(list(results.values()))
    expected_acceptance = {
        "both_complete_pass": all(result["completed"] for result in results.values()),
        "registered_interruption_pass": (
            not partial["completed"]
            and partial["current_terms"] == 2
            and results["resumed"]["resumed"]
            and partial["stage_sha256"]
            == _canonical_json_sha256(results["resumed"]["stages"][:2])
        ),
        "clean_source_boundary_pass": all(
            not result["source_construction"]["historical_checkpoint_used"]
            and not result["source_construction"]["ci_initializer_used"]
            for result in results.values()
        ),
        "candidate_reproduction_pass": comparison["selected_candidates"][
            "resumed"
        ]
        == comparison["selected_candidates"]["clean"],
        "clean_resume_energy_match_pass": max(differences)
        <= thresholds["resume_energy_tolerance"],
        "energy_nonincreasing_pass": all(
            right <= left + thresholds["energy_nesting_tolerance"]
            for lineage in rebuilt_energies.values()
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
        "command_resource_pass": max(
            execution["command_times_seconds"].values()
        )
        <= thresholds["command_wall_time_maximum_seconds"]
        and max(
            partial["peak_cpu_rss_bytes"],
            *(result["peak_cpu_rss_bytes"] for result in results.values()),
        )
        <= thresholds["peak_cpu_rss_maximum_bytes"],
        "external_cap_boundary_pass": all(
            result["automatic_stopping_rule"] == "not_admitted"
            and result["external_max_terms_required"]
            for result in results.values()
        ),
    }
    expected_acceptance["phase39_n4_d8_internal_pass"] = all(
        expected_acceptance.values()
    )
    if artifact["acceptance"] != expected_acceptance:
        raise AssertionError("restored Phase 39 acceptance record does not recompute")

    return {
        "verified": True,
        "accepted": expected_acceptance["phase39_n4_d8_internal_pass"],
        "rebuilt_lineages": rebuilt,
        "selected_candidates": comparison["selected_candidates"],
        "maximum_clean_resume_energy_difference": max(differences),
        "maximum_final_error_vs_CI": max(map(abs, final_errors)),
        "maximum_final_variance": max(final_variances),
        "d6_to_d8_final_energy_change": rebuilt["resumed"][-1]["energy"]
        - historical["final_energy"],
        "optimizer_failure_count": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path("docs/experiments/results/phase39_n4_d8_clean_source.json"),
    )
    arguments = parser.parse_args()
    print(json.dumps(verify_artifact(arguments.artifact), indent=2))


if __name__ == "__main__":
    main()
