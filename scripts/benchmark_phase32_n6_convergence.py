"""Run the registered Phase 32 N=6 independent D/K convergence experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch

from femps.algorithms import (
    DiagonalPathConfig,
    canonical_slater_orbitals,
    embed_diagonal_path_orbitals,
    extend_diagonal_path_terms,
    load_diagonal_path_checkpoint,
    run_diagonal_path_variable_projection,
    solve_generalized_hermitian,
)
from femps.benchmarks import ProcessRSSMonitor
from femps.exterior import (
    antisymmetry_residual,
    diagonal_path_exterior_coefficients,
    diagonal_path_hamiltonian_matrices,
    exterior_coefficients_to_tensor,
    particle_tt_ranks_exterior_coefficients,
)
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)


PARTICLES = 6
DIMENSIONS = (8, 10, 12)
TERMS_AXIS = (1, 2, 4)
QUADRATURE = 128
QUADRATURE_CHECK = 160
COUPLING = 1.0
SOFTENING = 1.0
SEEDS = {"D10_K1": 3201, "D10_K2": 3202, "D10_K4": 3204, "D8_K4": 3284, "D12_K4": 3212}
STEPS = 160
LBFGS_STEPS = 80
CPU_RSS_CAP_BYTES = 2 * 1024**3
GPU_MEMORY_CAP_BYTES = 4 * 1024**3
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


def _basis_audit(dimension: int) -> dict:
    started = time.perf_counter()
    with ProcessRSSMonitor() as monitor:
        one_body = harmonic_pair_hamiltonian(
            dimension, kappa=0.0, dtype=torch.complex128, device="cpu"
        )[0]
        dense_pair = soft_coulomb_dense_quadrature(
            dimension,
            quadrature_order=QUADRATURE,
            coupling=COUPLING,
            softening=SOFTENING,
            dtype=torch.complex128,
            device="cpu",
        )
        dense_pair_check = soft_coulomb_dense_quadrature(
            dimension,
            quadrature_order=QUADRATURE_CHECK,
            coupling=COUPLING,
            softening=SOFTENING,
            dtype=torch.complex128,
            device="cpu",
        )
        quadrature_change = torch.linalg.vector_norm(
            dense_pair - dense_pair_check
        ) / torch.linalg.vector_norm(dense_pair_check)
        interaction, diagnostics = soft_coulomb_operator(
            dimension,
            quadrature_order=QUADRATURE,
            coupling=COUPLING,
            softening=SOFTENING,
            relative_threshold=1e-13,
            factorization_backend="physical",
            dtype=torch.complex128,
            device="cpu",
        )
        hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
            one_body, PARTICLES, dense_pair
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
        ci_coefficients = eigenvectors[:, 0]
        ci_energy = eigenvalues[0].real
        ci_residual = hamiltonian @ ci_coefficients - ci_energy * ci_coefficients

        slater_coefficients = torch.zeros_like(ci_coefficients)
        slater_coefficients[0] = 1.0
        slater_acted = hamiltonian @ slater_coefficients
        slater_energy = torch.vdot(slater_coefficients, slater_acted).real
        slater_residual = slater_acted - slater_energy * slater_coefficients
        ci_ranks = particle_tt_ranks_exterior_coefficients(
            ci_coefficients, dimension, PARTICLES
        )
        slater_ranks = particle_tt_ranks_exterior_coefficients(
            slater_coefficients, dimension, PARTICLES
        )
    memory = monitor.record()
    return {
        "D": dimension,
        "one_body": one_body,
        "interaction": interaction,
        "dense_hamiltonian": hamiltonian,
        "public": {
            "D": dimension,
            "Q": QUADRATURE,
            "Q_check": QUADRATURE_CHECK,
            "quadrature_relative_change_Q128_vs_Q160": float(quadrature_change),
            "physical_operator_svd_rank": diagnostics.retained_rank,
            "physical_operator_svd_relative_error": (
                diagnostics.dense_relative_factorization_error
            ),
            "exterior_ci_dimension": math.comb(dimension, PARTICLES),
            "forbidden_particle_tensor_coefficients": dimension**PARTICLES,
            "direct_ci": {
                "energy": float(ci_energy),
                "energy_variance": float(torch.vdot(ci_residual, ci_residual).real),
                "norm_error": float(
                    abs(torch.vdot(ci_coefficients, ci_coefficients).real - 1.0)
                ),
                "ordinary_particle_tt_ranks": list(ci_ranks),
                "ordinary_particle_tt_storage_scalars": _tt_storage(
                    dimension, ci_ranks
                ),
                "raw_exterior_coefficients": _complex_vector(ci_coefficients),
            },
            "reference_slater": {
                "definition": "first N harmonic-oscillator orbitals",
                "energy": float(slater_energy),
                "energy_variance": float(
                    torch.vdot(slater_residual, slater_residual).real
                ),
                "norm_error": 0.0,
                "ordinary_particle_tt_ranks": list(slater_ranks),
                "ordinary_particle_tt_storage_scalars": _tt_storage(
                    dimension, slater_ranks
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_memory": memory.as_dict(),
        },
    }


def _config(dimension: int, terms: int, seed: int, device: str) -> DiagonalPathConfig:
    return DiagonalPathConfig(
        basis_order=dimension,
        particles=PARTICLES,
        terms=terms,
        interaction_model="soft_coulomb",
        soft_coulomb_coupling=COUPLING,
        soft_coulomb_softening=SOFTENING,
        soft_coulomb_quadrature_order=QUADRATURE,
        steps=STEPS,
        learning_rate=1e-3,
        final_learning_rate=1e-5,
        seed=seed,
        device=device,
        record_points=10,
        checkpoint_every=STEPS,
        lbfgs_refinement_steps=LBFGS_STEPS,
        truth_maximum_dimension=1,
        particle_tensor_maximum_coefficients=100_000,
    )


def _evaluate_checkpoint(
    result: dict,
    checkpoint: Path,
    basis: dict,
    *,
    admit_materialization: bool,
) -> dict:
    payload = load_diagonal_path_checkpoint(checkpoint)
    orbitals = canonical_slater_orbitals(payload["best_raw"])
    overlap, transition_hamiltonian = diagonal_path_hamiltonian_matrices(
        orbitals,
        basis["one_body"],
        two_body_left=basis["interaction"].left,
        two_body_right=basis["interaction"].right,
        two_body_weights=basis["interaction"].weights,
    )
    solved = solve_generalized_hermitian(
        transition_hamiltonian, overlap, relative_threshold=1e-10
    )
    coefficients = diagonal_path_exterior_coefficients(
        orbitals, solved.amplitudes
    )
    norm = torch.vdot(coefficients, coefficients).real
    acted = basis["dense_hamiltonian"] @ coefficients
    energy = (torch.vdot(coefficients, acted) / norm).real
    residual = acted - energy * coefficients
    variance = torch.vdot(residual, residual).real / norm
    ranks = particle_tt_ranks_exterior_coefficients(
        coefficients, result["config"]["basis_order"], PARTICLES
    )
    materialized = None
    if admit_materialization:
        particle_state = exterior_coefficients_to_tensor(
            coefficients, result["config"]["basis_order"], PARTICLES
        )
        materialized = float(antisymmetry_residual(particle_state))
    result.update(
        {
            "checkpoint_sha256": _sha256(checkpoint),
            "dense_quadrature_energy": float(energy),
            "dense_quadrature_energy_variance": float(variance),
            "dense_quadrature_norm_error": float(abs(norm - 1.0)),
            "error_vs_dense_quadrature_ci": (
                float(energy) - basis["public"]["direct_ci"]["energy"]
            ),
            "factorized_vs_dense_energy_difference": result["energy"]
            - float(energy),
            "validation_materialized_antisymmetry_residual": materialized,
            "ordinary_particle_tt_ranks_compact": list(ranks),
            "ordinary_particle_tt_storage_scalars": _tt_storage(
                result["config"]["basis_order"], ranks
            ),
            "femps_stored_parameter_scalars": (
                result["config"]["terms"]
                * result["config"]["basis_order"]
                * PARTICLES
                + result["config"]["terms"]
            ),
            "raw_exterior_coefficients": _complex_vector(coefficients),
        }
    )
    return result


def _run_point(
    point_id: str,
    dimension: int,
    terms: int,
    seed: int,
    device: str,
    checkpoint_dir: Path,
    basis: dict,
    *,
    initial_orbitals: torch.Tensor | None = None,
    initialization_lineage: dict,
    admit_materialization: bool = False,
) -> tuple[dict, Path]:
    checkpoint = checkpoint_dir / f"{point_id}.pt"
    result = run_diagonal_path_variable_projection(
        _config(dimension, terms, seed, device),
        checkpoint_path=checkpoint,
        initial_orbitals=initial_orbitals,
        operators=(basis["one_body"], basis["interaction"]),
        operator_id=f"soft_N6_D{dimension}_Q128_physical_svd",
    )
    result["point_id"] = point_id
    result["initialization_lineage"] = initialization_lineage
    _evaluate_checkpoint(
        result,
        checkpoint,
        basis,
        admit_materialization=admit_materialization,
    )
    print(
        point_id,
        result["dense_quadrature_energy"],
        result["error_vs_dense_quadrature_ci"],
        flush=True,
    )
    return result, checkpoint


def _axis_row(point: dict) -> dict:
    return {
        "point_id": point["point_id"],
        "D": point["config"]["basis_order"],
        "K": point["config"]["terms"],
        "energy": point["dense_quadrature_energy"],
        "error_vs_direct_ci": point["error_vs_dense_quadrature_ci"],
        "energy_variance": point["dense_quadrature_energy_variance"],
        "norm_error": point["dense_quadrature_norm_error"],
        "structural_antisymmetry_residual": point[
            "structural_antisymmetry_residual"
        ],
        "materialized_antisymmetry_residual": point[
            "validation_materialized_antisymmetry_residual"
        ],
        "retained_condition_number": point["retained_condition_number"],
        "elapsed_seconds": point["total_elapsed_seconds_this_call"],
        "peak_cpu_rss_bytes": point["peak_cpu_rss_bytes"],
        "peak_cuda_memory_bytes": point["peak_cuda_memory_bytes"],
        "ordinary_particle_tt_ranks": point[
            "ordinary_particle_tt_ranks_compact"
        ],
        "ordinary_particle_tt_storage_scalars": point[
            "ordinary_particle_tt_storage_scalars"
        ],
        "femps_stored_parameter_scalars": point["femps_stored_parameter_scalars"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase32_n6_convergence"),
    )
    parser.add_argument(
        "--resource-audit",
        type=Path,
        default=Path("docs/experiments/results/phase32_n6_resource_audit.json"),
    )
    parser.add_argument(
        "--stability-artifact",
        type=Path,
        default=Path(
            "docs/experiments/results/phase29_n6_multiseed_stability.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/phase32_n6_convergence.json"),
    )
    args = parser.parse_args()

    resource = json.loads(args.resource_audit.read_text(encoding="utf-8"))
    if not resource["D12_estimate"]["admitted_before_production"]:
        raise RuntimeError("D=12 production was not admitted by the written preflight")
    stability = json.loads(args.stability_artifact.read_text(encoding="utf-8"))
    if not stability["acceptance"]["multiseed_pass"]:
        raise RuntimeError("registered decisive-point multiseed evidence did not pass")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bases = {dimension: _basis_audit(dimension) for dimension in DIMENSIONS}

    k1, k1_checkpoint = _run_point(
        "N6_D10_K1_seed3201_blind",
        10,
        1,
        SEEDS["D10_K1"],
        args.device,
        args.checkpoint_dir,
        bases[10],
        initialization_lineage={
            "kind": "blind_slater",
            "seed": SEEDS["D10_K1"],
            "truth_state_used": False,
        },
    )
    initial_k2 = extend_diagonal_path_terms(
        load_diagonal_path_checkpoint(k1_checkpoint)["best_raw"],
        2,
        seed=SEEDS["D10_K2"],
    )
    k2, k2_checkpoint = _run_point(
        "N6_D10_K2_seed3202_from_K1",
        10,
        2,
        SEEDS["D10_K2"],
        args.device,
        args.checkpoint_dir,
        bases[10],
        initial_orbitals=initial_k2,
        initialization_lineage={
            "kind": "exact_K1_span_plus_one_seeded_blind_slater",
            "seed": SEEDS["D10_K2"],
            "truth_state_used": False,
        },
    )
    initial_k4 = extend_diagonal_path_terms(
        load_diagonal_path_checkpoint(k2_checkpoint)["best_raw"],
        4,
        seed=SEEDS["D10_K4"],
    )
    k4_d10, k4_d10_checkpoint = _run_point(
        "N6_D10_K4_seed3204_from_K2",
        10,
        4,
        SEEDS["D10_K4"],
        args.device,
        args.checkpoint_dir,
        bases[10],
        initial_orbitals=initial_k4,
        initialization_lineage={
            "kind": "exact_K2_span_plus_two_seeded_blind_slaters",
            "seed": SEEDS["D10_K4"],
            "truth_state_used": False,
        },
        admit_materialization=True,
    )
    k4_d8, _ = _run_point(
        "N6_D8_K4_seed3284_blind",
        8,
        4,
        SEEDS["D8_K4"],
        args.device,
        args.checkpoint_dir,
        bases[8],
        initialization_lineage={
            "kind": "four_seeded_blind_slaters",
            "seed": SEEDS["D8_K4"],
            "truth_state_used": False,
        },
        admit_materialization=True,
    )
    initial_d12 = embed_diagonal_path_orbitals(
        load_diagonal_path_checkpoint(k4_d10_checkpoint)["best_raw"], 12
    )
    k4_d12, _ = _run_point(
        "N6_D12_K4_seed3212_from_D10",
        12,
        4,
        SEEDS["D12_K4"],
        args.device,
        args.checkpoint_dir,
        bases[12],
        initial_orbitals=initial_d12,
        initialization_lineage={
            "kind": "exact_zero_padding_of_optimized_D10_state",
            "seed": SEEDS["D12_K4"],
            "truth_state_used": False,
        },
    )

    points = [k1, k2, k4_d10, k4_d8, k4_d12]
    k_axis = [_axis_row(point) for point in (k1, k2, k4_d10)]
    d_axis = [_axis_row(point) for point in (k4_d8, k4_d10, k4_d12)]
    structural_pass = all(
        point["completed"]
        and point["structural_antisymmetry_residual"] <= 1e-12
        and point["dense_quadrature_norm_error"] <= 1e-10
        and point["structural_counts"]["enumerated_virtual_paths"] == 0
        and point["structural_counts"]["materialized_particle_coefficients"] == 0
        for point in points
    )
    materialization_pass = all(
        point["validation_materialized_antisymmetry_residual"] <= 1e-12
        for point in (k4_d8, k4_d10)
    ) and k4_d12["validation_materialized_antisymmetry_residual"] is None
    k_axis_pass = all(
        k_axis[index + 1]["energy"] <= k_axis[index]["energy"] + 1e-9
        for index in range(len(k_axis) - 1)
    )
    d_axis_pass = all(
        d_axis[index + 1]["energy"] <= d_axis[index]["energy"] + 1e-9
        for index in range(len(d_axis) - 1)
    ) and all(
        bases[DIMENSIONS[index + 1]]["public"]["direct_ci"]["energy"]
        <= bases[DIMENSIONS[index]]["public"]["direct_ci"]["energy"] + 1e-10
        for index in range(len(DIMENSIONS) - 1)
    )
    operator_pass = all(
        basis["public"]["physical_operator_svd_relative_error"] <= 1e-11
        and basis["public"]["quadrature_relative_change_Q128_vs_Q160"] <= 2e-12
        for basis in bases.values()
    )
    resource_pass = all(
        point["total_elapsed_seconds_this_call"] <= WALL_TIME_CAP_SECONDS
        and point["peak_cpu_rss_bytes"] <= CPU_RSS_CAP_BYTES
        and (
            point["peak_cuda_memory_bytes"] is None
            or point["peak_cuda_memory_bytes"] <= GPU_MEMORY_CAP_BYTES
        )
        for point in points
    )
    accepted = bool(
        structural_pass
        and materialization_pass
        and k_axis_pass
        and d_axis_pass
        and operator_pass
        and resource_pass
    )
    artifact = {
        "schema_version": 1,
        "experiment": "phase32_N6_independent_D_K_convergence",
        "evidence_level": "numerical",
        "scientific_boundary": (
            "three D values and three K values for one soft-Coulomb model; "
            "convergence evidence, not an asymptotic scaling fit"
        ),
        "model": {
            "N": PARTICLES,
            "dimensions": list(DIMENSIONS),
            "terms": list(TERMS_AXIS),
            "Q": QUADRATURE,
            "Q_check": QUADRATURE_CHECK,
            "coupling": COUPLING,
            "softening": SOFTENING,
        },
        "registered_config": {
            "seeds": SEEDS,
            "steps": STEPS,
            "lbfgs_steps": LBFGS_STEPS,
            "learning_rate": 1e-3,
            "final_learning_rate": 1e-5,
            "device": args.device,
            "truth_state_initialization": False,
        },
        "source_records": {
            "resource_audit": {
                "path": args.resource_audit.as_posix(),
                "sha256": _sha256(args.resource_audit),
            },
            "decisive_K4_multiseed": {
                "path": args.stability_artifact.as_posix(),
                "sha256": _sha256(args.stability_artifact),
                "seeds": stability["seeds"],
                "energy_spread": stability["stability"]["energy_spread"],
                "pass": stability["acceptance"]["multiseed_pass"],
            },
        },
        "basis_audits": [bases[d]["public"] for d in DIMENSIONS],
        "points": points,
        "correlation_axis": k_axis,
        "basis_axis": d_axis,
        "thresholds": {
            "energy_monotonicity_tolerance": 1e-9,
            "norm_error": 1e-10,
            "antisymmetry_residual": 1e-12,
            "operator_factorization_error": 1e-11,
            "quadrature_relative_change": 2e-12,
            "peak_cpu_rss_bytes": CPU_RSS_CAP_BYTES,
            "peak_cuda_memory_bytes": GPU_MEMORY_CAP_BYTES,
            "wall_time_seconds_per_point": WALL_TIME_CAP_SECONDS,
        },
        "acceptance": {
            "structural_pass": structural_pass,
            "materialization_validation_pass": materialization_pass,
            "K_axis_pass": k_axis_pass,
            "D_axis_pass": d_axis_pass,
            "operator_pass": operator_pass,
            "resource_pass": resource_pass,
            "phase32_convergence_pass": accepted,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "cuda_device_name": (
                torch.cuda.get_device_name(torch.device(args.device))
                if torch.device(args.device).type == "cuda"
                else None
            ),
        },
    }
    _write(args.output, artifact)
    print(json.dumps({"output": str(args.output), "accepted": accepted}, indent=2))


if __name__ == "__main__":
    main()
