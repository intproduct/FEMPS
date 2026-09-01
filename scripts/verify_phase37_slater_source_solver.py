"""Independently verify the committed Phase 37 clean-source artifact."""

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

from femps.algorithms import (
    canonical_lowest_slater,
    load_slater_source_command_config,
    select_adaptive_diagonal_path_term,
    solve_generalized_hermitian,
    validate_slater_source_result,
)
from femps.exterior import (
    antisymmetry_residual,
    diagonal_path_exterior_coefficients,
    diagonal_path_hamiltonian_matrices,
    exterior_coefficients_to_tensor,
    particle_tt_ranks,
)
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)


def _text_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _decode(values: list) -> torch.Tensor:
    return torch.view_as_complex(torch.tensor(values, dtype=torch.float64).contiguous())


def _close(observed: float, expected: float, label: str, tolerance: float) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(f"{label}: {observed} != {expected}")


def verify_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["schema_version"] != 1 or artifact["evidence_level"] != "numerical":
        raise AssertionError("unsupported Phase 37 artifact schema/evidence label")
    for source, expected in artifact["source_hashes"].items():
        if _text_sha256(Path(source)) != expected:
            raise AssertionError(f"Phase 37 normalized source hash mismatch: {source}")
    phase28 = artifact["phase28_artifact"]
    if _text_sha256(Path(phase28["path"])) != phase28["normalized_text_sha256"]:
        raise AssertionError("Phase 28 comparator artifact hash mismatch")

    config_path = Path("docs/experiments/configs/phase37_n4_d6_k4.json")
    config, config_record = load_slater_source_command_config(config_path)
    if artifact["registered_config"] != config_record:
        raise AssertionError("registered Phase 37 configuration changed")
    resumed = artifact["resumed_result"]
    clean = artifact["clean_result"]
    validate_slater_source_result(resumed, require_completed=True)
    validate_slater_source_result(clean, require_completed=True)
    if resumed["source_construction"]["historical_checkpoint_used"]:
        raise AssertionError("historical FEMPS source was used")
    if resumed["source_construction"]["ci_initializer_used"]:
        raise AssertionError("CI source was used")

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
        "dense CI energy",
        1e-12,
    )
    _close(
        diagnostics.dense_relative_factorization_error,
        resumed["operator_metadata"]["dense_relative_factorization_error"],
        "operator factorization error",
        1e-15,
    )
    initial = canonical_lowest_slater(config)
    if initial.shape != (1, config.basis_order, config.particles):
        raise AssertionError("canonical initial Slater shape changed")

    orbitals = {
        int(record["terms"]): _decode(record["values"])
        for record in artifact["stage_orbitals"]
    }
    if sorted(orbitals) != list(range(1, config.max_terms + 1)):
        raise AssertionError("committed stage orbitals are incomplete")
    rebuilt = []
    previous = None
    for terms, stage in enumerate(resumed["stages"], start=1):
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
        _close(float(solved.energy), point["energy"], f"K{terms} factorized energy", 2e-11)
        _close(float(direct_energy), point["energy"], f"K{terms} direct energy", 2e-10)
        _close(float(abs(norm - 1)), point["norm_error"], f"K{terms} norm error", 2e-12)
        _close(float(variance), point["energy_variance"], f"K{terms} variance", 2e-9)
        if float(antisymmetry_residual(particle)) > 1e-12:
            raise AssertionError(f"K{terms} materialized antisymmetry failed")
        ranks = list(particle_tt_ranks(particle))
        if ranks != point["ordinary_particle_tt_ranks"]:
            raise AssertionError(f"K{terms} ordinary particle-TT ranks changed")
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
                raise AssertionError(f"K{terms} candidate selection changed")
            _close(
                growth.predicted_improvement,
                stage["predicted_improvement"],
                f"K{terms} predicted improvement",
                2e-11,
            )
        rebuilt.append(
            {
                "terms": terms,
                "energy": float(direct_energy),
                "error_vs_CI": float(direct_energy) - dense_ci_energy,
                "variance": float(variance),
                "norm_error": float(abs(norm - 1)),
                "ordinary_particle_tt_ranks": ranks,
            }
        )
        previous = current

    comparison = artifact["comparison"]
    clean_energies = [
        stage["optimizer_result"]["energy"] for stage in clean["stages"]
    ]
    resumed_energies = [
        stage["optimizer_result"]["energy"] for stage in resumed["stages"]
    ]
    differences = [
        abs(left - right)
        for left, right in zip(clean_energies, resumed_energies, strict=True)
    ]
    if differences != comparison["energy_absolute_differences"]:
        raise AssertionError("clean/resume energy differences changed")
    if max(differences) > config_record["acceptance"]["resume_energy_tolerance"]:
        raise AssertionError("clean/resume energy tolerance failed")
    if not all(artifact["acceptance"].values()):
        raise AssertionError("committed Phase 37 acceptance gate failed")
    return {
        "verified": True,
        "rebuilt_stages": rebuilt,
        "selected_candidates": comparison["resumed_candidates"],
        "maximum_clean_resume_energy_difference": max(differences),
        "final_error_vs_CI": rebuilt[-1]["error_vs_CI"],
        "automatic_stopping_rule": "not_admitted",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path(
            "docs/experiments/results/phase37_slater_source_solver.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), indent=2))


if __name__ == "__main__":
    main()
