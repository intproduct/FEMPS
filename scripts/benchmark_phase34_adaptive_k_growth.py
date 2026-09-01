"""Run the preregistered Phase 34 truth-free adaptive K-growth benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform

import torch

from femps.algorithms import (
    DiagonalPathConfig,
    assess_term_pruning,
    canonical_slater_orbitals,
    load_diagonal_path_checkpoint,
    run_diagonal_path_variable_projection,
    select_adaptive_diagonal_path_term,
    solve_generalized_hermitian,
)
from femps.exterior import (
    diagonal_path_exterior_coefficients,
    diagonal_path_hamiltonian_matrices,
    particle_tt_ranks_exterior_coefficients,
)
from femps.hamiltonians import (
    FactorizedTwoBodyOperator,
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)


PARTICLES = 6
DIMENSION = 12
QUADRATURE = 128
QUADRATURE_CHECK = 160
COUPLING = 1.0
SOFTENING = 1.0
POOL_SIZE = 32
GROWTH_SEEDS = {"K5": 3451, "K6": 3452}
COLD_K6_SEED = 3460
STEPS = 160
LBFGS_STEPS = 80
LEARNING_RATE = 1e-3
FINAL_LEARNING_RATE = 1e-5
OVERLAP_THRESHOLD = 1e-10
CONDITION_THRESHOLD = 1e8
ENERGY_MONOTONICITY_TOLERANCE = 1e-9
MEASURABLE_IMPROVEMENT = 1e-8
PRUNING_ENERGY_TOLERANCE = 1e-7
CPU_RSS_CAP_BYTES = 2 * 1024**3
WALL_TIME_CAP_SECONDS = 600.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _complex_vector(values: torch.Tensor) -> list[list[float]]:
    cpu = values.detach().to(dtype=torch.complex128, device="cpu")
    return [[float(value.real), float(value.imag)] for value in cpu]


def _tt_storage(dimension: int, ranks: tuple[int, ...]) -> int:
    extended = (1,) + ranks + (1,)
    return sum(
        extended[index] * dimension * extended[index + 1]
        for index in range(len(extended) - 1)
    )


def _config(terms: int, seed: int) -> DiagonalPathConfig:
    return DiagonalPathConfig(
        basis_order=DIMENSION,
        particles=PARTICLES,
        terms=terms,
        interaction_model="soft_coulomb",
        soft_coulomb_coupling=COUPLING,
        soft_coulomb_softening=SOFTENING,
        soft_coulomb_quadrature_order=QUADRATURE,
        steps=STEPS,
        learning_rate=LEARNING_RATE,
        final_learning_rate=FINAL_LEARNING_RATE,
        seed=seed,
        device="cpu",
        record_points=10,
        checkpoint_every=STEPS,
        overlap_relative_threshold=OVERLAP_THRESHOLD,
        lbfgs_refinement_steps=LBFGS_STEPS,
        truth_maximum_dimension=1,
        particle_tensor_maximum_coefficients=100_000,
    )


def _operators() -> tuple[torch.Tensor, FactorizedTwoBodyOperator, dict]:
    one_body = harmonic_pair_hamiltonian(
        DIMENSION, kappa=0.0, dtype=torch.complex128, device="cpu"
    )[0]
    interaction, diagnostics = soft_coulomb_operator(
        DIMENSION,
        quadrature_order=QUADRATURE,
        coupling=COUPLING,
        softening=SOFTENING,
        relative_threshold=1e-13,
        factorization_backend="physical",
        dtype=torch.complex128,
        device="cpu",
    )
    return one_body, interaction, {
        "physical_operator_svd_rank": diagnostics.retained_rank,
        "physical_operator_svd_relative_error": (
            diagnostics.dense_relative_factorization_error
        ),
    }


def _run_point(
    point_id: str,
    terms: int,
    seed: int,
    initial_orbitals: torch.Tensor | None,
    checkpoint_dir: Path,
    one_body: torch.Tensor,
    interaction: FactorizedTwoBodyOperator,
    lineage: dict,
) -> tuple[dict, torch.Tensor, Path]:
    checkpoint = checkpoint_dir / f"{point_id}.pt"
    result = run_diagonal_path_variable_projection(
        _config(terms, seed),
        checkpoint_path=checkpoint,
        initial_orbitals=initial_orbitals,
        operators=(one_body, interaction),
        operator_id="soft_N6_D12_Q128_physical_svd_phase34",
    )
    payload = load_diagonal_path_checkpoint(checkpoint)
    orbitals = canonical_slater_orbitals(payload["best_raw"])
    result["point_id"] = point_id
    result["initialization_lineage"] = lineage
    result["checkpoint_sha256"] = _sha256(checkpoint)
    print(point_id, result["energy"], flush=True)
    return result, orbitals, checkpoint


def _growth_record(growth: object) -> dict:
    return {
        "seed": growth.seed,
        "pool_size": growth.pool_size,
        "source_terms": growth.source_terms,
        "selected_candidate": growth.selected_candidate,
        "source_energy": growth.source_energy,
        "predicted_energy": growth.predicted_energy,
        "predicted_improvement": growth.predicted_improvement,
        "truth_state_used": False,
        "ranking_inputs": [
            "factorized determinant-transition Hamiltonian",
            "overlap matrix",
            "generalized-eigenvalue energy",
            "balanced overlap conditioning",
        ],
        "candidates": [asdict(candidate) for candidate in growth.candidates],
    }


def _truth_data(one_body: torch.Tensor) -> tuple[torch.Tensor, dict]:
    dense_pair = soft_coulomb_dense_quadrature(
        DIMENSION,
        quadrature_order=QUADRATURE,
        coupling=COUPLING,
        softening=SOFTENING,
        dtype=torch.complex128,
        device="cpu",
    )
    dense_pair_check = soft_coulomb_dense_quadrature(
        DIMENSION,
        quadrature_order=QUADRATURE_CHECK,
        coupling=COUPLING,
        softening=SOFTENING,
        dtype=torch.complex128,
        device="cpu",
    )
    quadrature_change = torch.linalg.vector_norm(
        dense_pair - dense_pair_check
    ) / torch.linalg.vector_norm(dense_pair_check)
    dense_hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, PARTICLES, dense_pair
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(dense_hamiltonian)
    ci_coefficients = eigenvectors[:, 0]
    ci_energy = eigenvalues[0].real
    ci_residual = dense_hamiltonian @ ci_coefficients - ci_energy * ci_coefficients
    ci_ranks = particle_tt_ranks_exterior_coefficients(
        ci_coefficients, DIMENSION, PARTICLES
    )
    slater_coefficients = torch.zeros_like(ci_coefficients)
    slater_coefficients[0] = 1.0
    slater_acted = dense_hamiltonian @ slater_coefficients
    slater_energy = torch.vdot(slater_coefficients, slater_acted).real
    slater_residual = slater_acted - slater_energy * slater_coefficients
    slater_ranks = particle_tt_ranks_exterior_coefficients(
        slater_coefficients, DIMENSION, PARTICLES
    )
    return dense_hamiltonian, {
        "exterior_ci_dimension": math.comb(DIMENSION, PARTICLES),
        "forbidden_particle_tensor_coefficients": DIMENSION**PARTICLES,
        "quadrature_relative_change_Q128_vs_Q160": float(quadrature_change),
        "direct_ci": {
            "energy": float(ci_energy),
            "energy_variance": float(torch.vdot(ci_residual, ci_residual).real),
            "norm_error": float(
                abs(torch.vdot(ci_coefficients, ci_coefficients).real - 1.0)
            ),
            "ordinary_particle_tt_ranks": list(ci_ranks),
            "ordinary_particle_tt_storage_scalars": _tt_storage(
                DIMENSION, ci_ranks
            ),
        },
        "reference_slater": {
            "energy": float(slater_energy),
            "energy_variance": float(
                torch.vdot(slater_residual, slater_residual).real
            ),
            "norm_error": 0.0,
            "ordinary_particle_tt_ranks": list(slater_ranks),
            "ordinary_particle_tt_storage_scalars": _tt_storage(
                DIMENSION, slater_ranks
            ),
        },
    }


def _audit_state(
    point: dict,
    orbitals: torch.Tensor,
    one_body: torch.Tensor,
    interaction: FactorizedTwoBodyOperator,
    dense_hamiltonian: torch.Tensor,
    ci_energy: float,
) -> dict:
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        orbitals,
        one_body,
        two_body_left=interaction.left,
        two_body_right=interaction.right,
        two_body_weights=interaction.weights,
    )
    solved = solve_generalized_hermitian(
        hamiltonian, overlap, relative_threshold=OVERLAP_THRESHOLD
    )
    coefficients = diagonal_path_exterior_coefficients(
        orbitals, solved.amplitudes
    )
    norm = torch.vdot(coefficients, coefficients).real
    acted = dense_hamiltonian @ coefficients
    energy = (torch.vdot(coefficients, acted) / norm).real
    residual = acted - energy * coefficients
    variance = torch.vdot(residual, residual).real / norm
    ranks = particle_tt_ranks_exterior_coefficients(
        coefficients, DIMENSION, PARTICLES
    )
    point.update(
        {
            "dense_quadrature_energy": float(energy),
            "dense_quadrature_energy_variance": float(variance),
            "dense_quadrature_norm_error": float(abs(norm - 1.0)),
            "error_vs_direct_ci": float(energy) - ci_energy,
            "factorized_vs_dense_energy_difference": point["energy"] - float(energy),
            "ordinary_particle_tt_ranks_compact": list(ranks),
            "ordinary_particle_tt_storage_scalars": _tt_storage(DIMENSION, ranks),
            "femps_stored_parameter_scalars": (
                point["config"]["terms"] * DIMENSION * PARTICLES
                + point["config"]["terms"]
            ),
            "raw_exterior_coefficients": _complex_vector(coefficients),
        }
    )
    return point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase32-artifact",
        type=Path,
        default=Path("docs/experiments/results/phase32_n6_convergence.json"),
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
        default=Path("checkpoints/phase34_adaptive_k_growth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/phase34_adaptive_k_growth.json"),
    )
    args = parser.parse_args()

    phase32 = json.loads(args.phase32_artifact.read_text(encoding="utf-8"))
    if not phase32["acceptance"]["phase32_convergence_pass"]:
        raise RuntimeError("Phase 32 source convergence gate did not pass")
    source_point = next(
        point
        for point in phase32["points"]
        if point["point_id"] == "N6_D12_K4_seed3212_from_D10"
    )
    if _sha256(args.source_checkpoint) != source_point["checkpoint_sha256"]:
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
    if abs(float(source_solved.energy) - source_point["energy"]) > 1e-10:
        raise RuntimeError("Phase 32 source state does not reproduce its energy")

    # Selection and nonlinear optimization finish before any dense CI object is
    # constructed.  Only factorized transition data are available here.
    growth_k5 = select_adaptive_diagonal_path_term(
        source_orbitals,
        one_body,
        interaction,
        pool_size=POOL_SIZE,
        seed=GROWTH_SEEDS["K5"],
        overlap_relative_threshold=OVERLAP_THRESHOLD,
        condition_threshold=CONDITION_THRESHOLD,
    )
    k5, k5_orbitals, _ = _run_point(
        "N6_D12_K5_seed3451_adaptive",
        5,
        GROWTH_SEEDS["K5"],
        growth_k5.orbitals,
        args.checkpoint_dir,
        one_body,
        interaction,
        {"kind": "truth_free_adaptive_K4_plus_one", **_growth_record(growth_k5)},
    )
    growth_k6 = select_adaptive_diagonal_path_term(
        k5_orbitals,
        one_body,
        interaction,
        pool_size=POOL_SIZE,
        seed=GROWTH_SEEDS["K6"],
        overlap_relative_threshold=OVERLAP_THRESHOLD,
        condition_threshold=CONDITION_THRESHOLD,
    )
    k6, k6_orbitals, _ = _run_point(
        "N6_D12_K6_seed3452_adaptive",
        6,
        GROWTH_SEEDS["K6"],
        growth_k6.orbitals,
        args.checkpoint_dir,
        one_body,
        interaction,
        {"kind": "truth_free_adaptive_K5_plus_one", **_growth_record(growth_k6)},
    )
    cold_k6, cold_k6_orbitals, _ = _run_point(
        "N6_D12_K6_seed3460_cold",
        6,
        COLD_K6_SEED,
        None,
        args.checkpoint_dir,
        one_body,
        interaction,
        {
            "kind": "cold_slater_plus_five_seeded_blind_slaters",
            "seed": COLD_K6_SEED,
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

    # Dense truth begins only after every adaptive/cold optimization is frozen.
    dense_hamiltonian, controls = _truth_data(one_body)
    ci_energy = controls["direct_ci"]["energy"]
    source = _audit_state(
        dict(source_point),
        source_orbitals,
        one_body,
        interaction,
        dense_hamiltonian,
        ci_energy,
    )
    k5 = _audit_state(
        k5, k5_orbitals, one_body, interaction, dense_hamiltonian, ci_energy
    )
    k6 = _audit_state(
        k6, k6_orbitals, one_body, interaction, dense_hamiltonian, ci_energy
    )
    cold_k6 = _audit_state(
        cold_k6,
        cold_k6_orbitals,
        one_body,
        interaction,
        dense_hamiltonian,
        ci_energy,
    )

    adaptive_points = [source, k5, k6]
    optimized_points = [k5, k6, cold_k6]
    monotonic_pass = all(
        adaptive_points[index + 1]["dense_quadrature_energy"]
        <= adaptive_points[index]["dense_quadrature_energy"]
        + ENERGY_MONOTONICITY_TOLERANCE
        for index in range(2)
    )
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
    selection_pass = all(
        growth.predicted_energy <= growth.source_energy + 1e-10
        and growth.candidates[growth.selected_candidate].admitted
        and growth.predicted_improvement >= -1e-10
        for growth in (growth_k5, growth_k6)
    )
    resource_pass = all(
        point["total_elapsed_seconds_this_call"] <= WALL_TIME_CAP_SECONDS
        and point["peak_cpu_rss_bytes"] <= CPU_RSS_CAP_BYTES
        for point in optimized_points
    )
    operator_pass = (
        operator_diagnostics["physical_operator_svd_relative_error"] <= 1e-11
        and controls["quadrature_relative_change_Q128_vs_Q160"] <= 2e-12
    )
    truth_pass = all(
        abs(point["factorized_vs_dense_energy_difference"]) <= 1e-10
        for point in optimized_points
    )
    pruning_pass = not pruning.should_prune
    accepted = bool(
        monotonic_pass
        and structural_pass
        and selection_pass
        and resource_pass
        and operator_pass
        and truth_pass
        and pruning_pass
    )

    artifact = {
        "schema_version": 1,
        "experiment": "phase34_truth_free_adaptive_N6_D12_K_growth",
        "evidence_level": "numerical",
        "scientific_boundary": (
            "one N6,D12 soft-Coulomb K4-to-K6 continuation and one cold K6 "
            "control; no N or asymptotic scaling claim"
        ),
        "registered_config": {
            "N": PARTICLES,
            "D": DIMENSION,
            "K_axis": [4, 5, 6],
            "Q": QUADRATURE,
            "Q_check": QUADRATURE_CHECK,
            "pool_size": POOL_SIZE,
            "growth_seeds": GROWTH_SEEDS,
            "cold_K6_seed": COLD_K6_SEED,
            "steps": STEPS,
            "lbfgs_steps": LBFGS_STEPS,
            "learning_rate": LEARNING_RATE,
            "final_learning_rate": FINAL_LEARNING_RATE,
            "truth_state_initialization": False,
            "truth_constructed_after_optimization": True,
        },
        "source_records": {
            "phase32_artifact": {
                "path": args.phase32_artifact.as_posix(),
                "sha256": _sha256(args.phase32_artifact),
            },
            "phase32_K4_checkpoint_sha256": _sha256(args.source_checkpoint),
            "source_hashes": {
                "growth": _sha256(
                    Path("src/femps/algorithms/diagonal_path_growth.py")
                ),
                "training": _sha256(
                    Path("src/femps/algorithms/diagonal_path_training.py")
                ),
                "benchmark": _sha256(Path(__file__)),
            },
        },
        "operator": {**operator_diagnostics, **controls},
        "growth": {
            "K4_to_K5": _growth_record(growth_k5),
            "K5_to_K6": _growth_record(growth_k6),
        },
        "adaptive_points": adaptive_points,
        "cold_K6_control": cold_k6,
        "pruning_assessment": asdict(pruning),
        "comparison": {
            "K5_energy_improvement": (
                source["dense_quadrature_energy"] - k5["dense_quadrature_energy"]
            ),
            "K6_energy_improvement_over_K5": (
                k5["dense_quadrature_energy"] - k6["dense_quadrature_energy"]
            ),
            "K6_total_energy_improvement_over_K4": (
                source["dense_quadrature_energy"] - k6["dense_quadrature_energy"]
            ),
            "K6_variance_change_from_K4": (
                k6["dense_quadrature_energy_variance"]
                - source["dense_quadrature_energy_variance"]
            ),
            "adaptive_K6_minus_cold_K6_energy": (
                k6["dense_quadrature_energy"]
                - cold_k6["dense_quadrature_energy"]
            ),
            "adaptive_growth_measurable": (
                source["dense_quadrature_energy"]
                - k6["dense_quadrature_energy"]
                >= MEASURABLE_IMPROVEMENT
            ),
        },
        "thresholds": {
            "energy_monotonicity_tolerance": ENERGY_MONOTONICITY_TOLERANCE,
            "measurable_improvement": MEASURABLE_IMPROVEMENT,
            "norm_error": 1e-10,
            "antisymmetry_residual": 1e-12,
            "retained_condition_number": CONDITION_THRESHOLD,
            "factorized_vs_dense_energy": 1e-10,
            "operator_factorization_error": 1e-11,
            "quadrature_relative_change": 2e-12,
            "pruning_energy_tolerance": PRUNING_ENERGY_TOLERANCE,
            "peak_cpu_rss_bytes": CPU_RSS_CAP_BYTES,
            "wall_time_seconds_per_point": WALL_TIME_CAP_SECONDS,
        },
        "acceptance": {
            "selection_pass": selection_pass,
            "K_axis_monotonic_pass": monotonic_pass,
            "structural_pass": structural_pass,
            "truth_reconstruction_pass": truth_pass,
            "operator_pass": operator_pass,
            "resource_pass": resource_pass,
            "pruning_not_triggered_pass": pruning_pass,
            "phase34_adaptive_growth_pass": accepted,
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
                "K4_energy": source["dense_quadrature_energy"],
                "K5_energy": k5["dense_quadrature_energy"],
                "K6_energy": k6["dense_quadrature_energy"],
                "cold_K6_energy": cold_k6["dense_quadrature_energy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
