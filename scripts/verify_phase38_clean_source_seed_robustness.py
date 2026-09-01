"""Independently verify the Phase 38 clean-source robustness artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from femps.algorithms import (  # noqa: E402
    canonical_lowest_slater,
    load_slater_source_command_config,
    select_adaptive_diagonal_path_term,
    solve_generalized_hermitian,
    validate_slater_source_result,
)
from femps.exterior import (  # noqa: E402
    antisymmetry_residual,
    diagonal_path_exterior_coefficients,
    diagonal_path_hamiltonian_matrices,
    exterior_coefficients_to_tensor,
    particle_tt_ranks,
)
from femps.hamiltonians import (  # noqa: E402
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)
from scripts.verify_phase37_slater_source_solver import (  # noqa: E402
    verify_artifact as verify_phase37_artifact,
)


def _text_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _decode(values: list) -> torch.Tensor:
    return torch.view_as_complex(torch.tensor(values, dtype=torch.float64).contiguous())


def _close(observed: float, expected: float, label: str, tolerance: float) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(f"{label}: {observed} != {expected}")


def _hash_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _operator_hash(one_body: torch.Tensor, interaction: object) -> str:
    digest = hashlib.sha256()
    for tensor in (
        one_body,
        interaction.left,
        interaction.right,
        interaction.weights,
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _rebuild_lineage(
    name: str,
    result: dict,
    orbital_records: list,
    config: object,
    one_body: torch.Tensor,
    interaction: object,
    dense_hamiltonian: torch.Tensor,
    dense_ci_energy: float,
) -> list[dict]:
    validate_slater_source_result(result, require_completed=True)
    initial = canonical_lowest_slater(config)
    if result["initial_source_identity"]["orbitals_sha256"] != _hash_tensor(initial):
        raise AssertionError(f"{name} canonical initial-source identity changed")
    if result["operator_identity"]["operator_sha256"] != _operator_hash(
        one_body, interaction
    ):
        raise AssertionError(f"{name} operator identity changed")
    orbitals = {
        int(record["terms"]): _decode(record["values"])
        for record in orbital_records
    }
    if sorted(orbitals) != [1, 2, 3, 4]:
        raise AssertionError(f"{name} committed stage orbitals are incomplete")

    rebuilt = []
    previous = None
    for terms, stage in enumerate(result["stages"], start=1):
        current = orbitals[terms]
        overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
            current,
            one_body,
            two_body_left=interaction.left,
            two_body_right=interaction.right,
            two_body_weights=interaction.weights,
        )
        solved = solve_generalized_hermitian(
            hamiltonian,
            overlap,
            relative_threshold=config.overlap_relative_threshold,
        )
        coefficients = diagonal_path_exterior_coefficients(
            current, solved.amplitudes
        )
        norm = torch.vdot(coefficients, coefficients).real
        direct_energy = torch.vdot(
            coefficients, dense_hamiltonian @ coefficients
        ).real / norm
        residual = dense_hamiltonian @ coefficients - direct_energy * coefficients
        variance = torch.vdot(residual, residual).real / norm
        particle = exterior_coefficients_to_tensor(
            coefficients, config.basis_order, config.particles
        )
        point = stage["optimizer_result"]
        _close(float(solved.energy), point["energy"], f"{name} K{terms} factorized", 2e-11)
        _close(float(direct_energy), point["energy"], f"{name} K{terms} direct", 2e-10)
        _close(float(abs(norm - 1)), point["norm_error"], f"{name} K{terms} norm", 2e-12)
        _close(float(variance), point["energy_variance"], f"{name} K{terms} variance", 2e-9)
        materialized_residual = float(antisymmetry_residual(particle))
        if materialized_residual > 1e-12:
            raise AssertionError(f"{name} K{terms} materialized antisymmetry failed")
        ranks = list(particle_tt_ranks(particle))
        if ranks != point["ordinary_particle_tt_ranks"]:
            raise AssertionError(f"{name} K{terms} particle-TT ranks changed")
        if terms > 1:
            stage_config = config.stages[terms - 2]
            growth = select_adaptive_diagonal_path_term(
                previous,
                one_body,
                interaction,
                pool_size=config.pool_size,
                seed=stage_config.candidate_seed,
                overlap_relative_threshold=config.overlap_relative_threshold,
                condition_threshold=config.condition_threshold,
                energy_nesting_tolerance=config.energy_nesting_tolerance,
            )
            if growth.selected_candidate != stage["selected_candidate"]:
                raise AssertionError(f"{name} K{terms} candidate selection changed")
            _close(
                growth.predicted_improvement,
                stage["predicted_improvement"],
                f"{name} K{terms} predicted improvement",
                2e-11,
            )
        rebuilt.append(
            {
                "terms": terms,
                "energy": float(direct_energy),
                "error_vs_CI": float(direct_energy) - dense_ci_energy,
                "variance": float(variance),
                "norm_error": float(abs(norm - 1)),
                "materialized_antisymmetry_residual": materialized_residual,
                "ordinary_particle_tt_ranks": ranks,
            }
        )
        previous = current
    return rebuilt


def verify_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1:
        raise AssertionError("unsupported Phase 38 artifact schema")
    if artifact.get("evidence_level") != "numerical":
        raise AssertionError("Phase 38 must remain numerical evidence")
    if artifact.get("experiment") != "phase38_clean_source_seed_robustness":
        raise AssertionError("unexpected Phase 38 experiment")
    for source, expected in artifact["source_hashes"].items():
        if _text_sha256(Path(source)) != expected:
            raise AssertionError(f"Phase 38 normalized source hash mismatch: {source}")

    config_paths = {
        "a": Path("docs/experiments/configs/phase38_n4_d6_k4_seed_a.json"),
        "b": Path("docs/experiments/configs/phase38_n4_d6_k4_seed_b.json"),
    }
    loaded = {
        name: load_slater_source_command_config(config_path)
        for name, config_path in config_paths.items()
    }
    for name, (_, record) in loaded.items():
        if record != artifact["registered_configs"][name]:
            raise AssertionError(f"registered Phase 38 config {name} changed")
    config_a, record_a = loaded["a"]
    config_b, record_b = loaded["b"]
    if (
        config_a.source_optimizer.seed,
        [(s.candidate_seed, s.optimizer_seed) for s in config_a.stages],
    ) != (3801, [(3811, 3812), (3821, 3822), (3831, 3832)]):
        raise AssertionError("schedule A changed")
    if (
        config_b.source_optimizer.seed,
        [(s.candidate_seed, s.optimizer_seed) for s in config_b.stages],
    ) != (3901, [(3911, 3912), (3921, 3922), (3931, 3932)]):
        raise AssertionError("schedule B changed")
    if record_a["acceptance"] != record_b["acceptance"]:
        raise AssertionError("Phase 38 acceptance records differ")
    thresholds = record_a["acceptance"]

    phase37_record = artifact["phase37_control"]
    phase37_path = Path(phase37_record["path"])
    if _text_sha256(phase37_path) != phase37_record["normalized_text_sha256"]:
        raise AssertionError("Phase 37 control hash changed")
    phase37_verified = verify_phase37_artifact(phase37_path)
    _close(
        phase37_verified["rebuilt_stages"][-1]["energy"],
        phase37_record["final_energy"],
        "Phase 37 final energy",
        2e-10,
    )

    one_body = harmonic_pair_hamiltonian(
        config_a.basis_order,
        kappa=0.0,
        omega=config_a.omega,
        dtype=torch.complex128,
        device="cpu",
    )[0]
    interaction, diagnostics = soft_coulomb_operator(
        config_a.basis_order,
        quadrature_order=config_a.quadrature_order,
        coupling=config_a.coupling,
        softening=config_a.softening,
        relative_threshold=config_a.relative_factor_threshold,
        factorization_backend=config_a.factorization_backend,
        dtype=torch.complex128,
        device="cpu",
    )
    dense_pair = soft_coulomb_dense_quadrature(
        config_a.basis_order,
        quadrature_order=config_a.quadrature_order,
        coupling=config_a.coupling,
        softening=config_a.softening,
        dtype=torch.complex128,
        device="cpu",
    )
    dense_hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, config_a.particles, dense_pair
    )
    dense_ci_energy = float(torch.linalg.eigvalsh(dense_hamiltonian)[0].real)
    _close(
        dense_ci_energy,
        artifact["dense_ci_comparator"]["energy"],
        "dense CI energy",
        1e-12,
    )
    if diagnostics.dense_relative_factorization_error > thresholds[
        "factorization_error_maximum"
    ]:
        raise AssertionError("operator factorization gate failed")

    results = artifact["results"]
    configs = {"a_resumed": config_a, "a_clean": config_a, "b_clean": config_b}
    rebuilt = {
        name: _rebuild_lineage(
            name,
            results[name],
            artifact["stage_orbitals"][name],
            configs[name],
            one_body,
            interaction,
            dense_hamiltonian,
            dense_ci_energy,
        )
        for name in ("a_resumed", "a_clean", "b_clean")
    }

    comparison = artifact["comparison"]
    rebuilt_energies = {
        name: [point["energy"] for point in points]
        for name, points in rebuilt.items()
    }
    for name, energies in rebuilt_energies.items():
        for observed, expected in zip(
            energies, comparison["energies"][name], strict=True
        ):
            _close(observed, expected, f"{name} serialized energy", 2e-10)
    a_differences = [
        abs(left - right)
        for left, right in zip(
            comparison["energies"]["a_resumed"],
            comparison["energies"]["a_clean"],
            strict=True,
        )
    ]
    if a_differences != comparison[
        "a_clean_resume_energy_absolute_differences"
    ]:
        raise AssertionError("A clean/resume differences changed")
    if max(a_differences) > thresholds["resume_energy_tolerance"]:
        raise AssertionError("A clean/resume tolerance failed")
    if comparison["selected_candidates"]["a_resumed"] != comparison[
        "selected_candidates"
    ]["a_clean"]:
        raise AssertionError("A selected candidates changed across resume")

    partial = artifact["execution"]["partial_a"]
    if partial["completed"] or partial["current_terms"] != 2:
        raise AssertionError("registered K2 interruption changed")
    if partial["stage_sha256"] != _canonical_json_sha256(
        results["a_resumed"]["stages"][:2]
    ):
        raise AssertionError("partial K2 record does not match resumed lineage")
    if len(artifact["execution"]["public_commands"]) != 4:
        raise AssertionError("unexpected Phase 38 command count")
    if artifact["execution"]["outcome_dependent_retries"] != 0:
        raise AssertionError("outcome-dependent retry was recorded")
    if not artifact["execution"]["truth_opened_after_all_fresh_optimizations"]:
        raise AssertionError("truth timing boundary failed")

    primary = ("a_resumed", "b_clean")
    final_errors = [rebuilt[name][-1]["error_vs_CI"] for name in primary]
    final_variances = [rebuilt[name][-1]["variance"] for name in primary]
    source_errors = [rebuilt[name][0]["error_vs_CI"] for name in primary]
    source_variances = [rebuilt[name][0]["variance"] for name in primary]
    phase37_final = phase37_record["final_energy"]
    combined_final = [phase37_final, *[rebuilt[name][-1]["energy"] for name in primary]]
    spread = max(combined_final) - min(combined_final)
    phase37_differences = [
        abs(rebuilt[name][-1]["energy"] - phase37_final) for name in primary
    ]
    all_points = [
        stage["optimizer_result"]
        for result in results.values()
        for stage in result["stages"]
    ]
    failures = sum(
        not point["completed"] or not point["refinement"]["accepted"]
        for point in all_points
    )
    command_times = artifact["execution"]["command_times_seconds"]
    expected_acceptance = {
        "all_complete_pass": all(result["completed"] for result in results.values()),
        "registered_interruption_pass": (
            not partial["completed"]
            and partial["current_terms"] == 2
            and results["a_resumed"]["resumed"]
            and partial["stage_sha256"]
            == _canonical_json_sha256(results["a_resumed"]["stages"][:2])
        ),
        "clean_source_boundary_pass": all(
            not result["source_construction"]["historical_checkpoint_used"]
            and not result["source_construction"]["ci_initializer_used"]
            for result in results.values()
        ),
        "a_candidate_reproduction_pass": comparison["selected_candidates"][
            "a_resumed"
        ] == comparison["selected_candidates"]["a_clean"],
        "a_clean_resume_energy_match_pass": max(a_differences)
        <= thresholds["resume_energy_tolerance"],
        "energy_nonincreasing_pass": all(
            right <= left + thresholds["energy_nesting_tolerance"]
            for lineage in rebuilt_energies.values()
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
        "final_energy_spread_pass": spread
        <= thresholds["final_energy_spread_maximum"],
        "phase37_difference_pass": max(phase37_differences)
        <= thresholds["fresh_difference_from_phase37_maximum"],
        "optimizer_failure_count_pass": failures
        <= thresholds["optimizer_failure_count_maximum"],
        "stage_scientific_records_pass": all(
            point["norm_error"] <= thresholds["norm_error_maximum"]
            and point["structural_antisymmetry_residual"]
            <= thresholds["antisymmetry_residual_maximum"]
            and point["materialized_antisymmetry_residual"]
            <= thresholds["antisymmetry_residual_maximum"]
            and point["structural_counts"]["enumerated_virtual_paths"] == 0
            and point["structural_counts"]["materialized_particle_coefficients"] == 0
            and point["total_elapsed_seconds_this_call"]
            <= thresholds["stage_wall_time_maximum_seconds"]
            and point["peak_cpu_rss_bytes"]
            <= thresholds["peak_cpu_rss_maximum_bytes"]
            for point in all_points
        ),
        "operator_factorization_pass": all(
            result["operator_metadata"]["dense_relative_factorization_error"]
            <= thresholds["factorization_error_maximum"]
            for result in results.values()
        ),
        "command_resource_pass": max(command_times.values())
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
    expected_acceptance["phase38_clean_source_seed_robustness_pass"] = all(
        expected_acceptance.values()
    )
    if artifact["acceptance"] != expected_acceptance:
        raise AssertionError("Phase 38 acceptance record does not recompute")
    if not expected_acceptance["phase38_clean_source_seed_robustness_pass"]:
        raise AssertionError("committed Phase 38 acceptance gate failed")

    return {
        "verified": True,
        "rebuilt_lineages": rebuilt,
        "selected_candidates": comparison["selected_candidates"],
        "maximum_a_clean_resume_energy_difference": max(a_differences),
        "combined_final_energy_spread": spread,
        "maximum_final_error_vs_CI": max(map(abs, final_errors)),
        "maximum_final_variance": max(final_variances),
        "optimizer_failure_count": failures,
        "automatic_stopping_rule": "not_admitted",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path(
            "docs/experiments/results/phase38_clean_source_seed_robustness.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
