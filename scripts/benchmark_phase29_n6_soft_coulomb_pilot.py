"""Run the ADR-0018 resource-capped N=6,D=10 soft-Coulomb pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

import torch

from femps.algorithms import (
    DiagonalPathConfig,
    extend_diagonal_path_terms,
    load_diagonal_path_checkpoint,
    run_diagonal_path_variable_projection,
)
from femps.benchmarks import ProcessRSSMonitor
from femps.exterior import (
    antisymmetry_residual,
    exterior_coefficients_to_tensor,
    particle_tt_ranks,
)
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)


PARTICLES = 6
DIMENSION = 10
QUADRATURE = 128
COUPLING = 1.0
SOFTENING = 1.0
K1_SEED = 29
K4_GROWTH_SEED = 2914
MEMORY_LIMIT_BYTES = 1_610_612_736
TIME_LIMIT_SECONDS = 600.0
FACTORIZATION_TOLERANCE = 1e-11
NORM_TOLERANCE = 1e-10
ANTISYMMETRY_TOLERANCE = 1e-12
K4_ERROR_TOLERANCE = 5e-4
K4_VARIANCE_TOLERANCE = 5e-3


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _dense_ci() -> dict:
    started = time.perf_counter()
    with ProcessRSSMonitor() as monitor:
        one_body = harmonic_pair_hamiltonian(
            DIMENSION, kappa=0.0, dtype=torch.complex128, device="cpu"
        )[0]
        dense_pair = soft_coulomb_dense_quadrature(
            DIMENSION,
            quadrature_order=QUADRATURE,
            coupling=COUPLING,
            softening=SOFTENING,
            dtype=torch.complex128,
            device="cpu",
        )
        hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
            one_body, PARTICLES, dense_pair
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
        energy = eigenvalues[0].real
        coefficients = eigenvectors[:, 0]
        residual = hamiltonian @ coefficients - energy * coefficients
        particle_state = exterior_coefficients_to_tensor(
            coefficients, DIMENSION, PARTICLES
        )
        symmetry = antisymmetry_residual(particle_state)
        ranks = particle_tt_ranks(particle_state)
    memory = monitor.record()
    return {
        "method": "dense_quadrature_finite_basis_ci",
        "role": "bounded independent truth, not FEMPS production",
        "N": PARTICLES,
        "D": DIMENSION,
        "Q": QUADRATURE,
        "energy": float(energy),
        "energy_variance": float(torch.vdot(residual, residual).real),
        "norm_error": float(abs(torch.vdot(coefficients, coefficients).real - 1.0)),
        "antisymmetry_residual": float(symmetry),
        "ordinary_particle_tt_ranks": list(ranks),
        "exterior_dimension": math.comb(DIMENSION, PARTICLES),
        "materialized_particle_coefficients": DIMENSION**PARTICLES,
        "elapsed_seconds": time.perf_counter() - started,
        "cpu_memory": memory.as_dict(),
    }


def _config(*, terms: int, steps: int, lbfgs_steps: int) -> DiagonalPathConfig:
    return DiagonalPathConfig(
        basis_order=DIMENSION,
        particles=PARTICLES,
        terms=terms,
        interaction_model="soft_coulomb",
        soft_coulomb_coupling=COUPLING,
        soft_coulomb_softening=SOFTENING,
        soft_coulomb_quadrature_order=QUADRATURE,
        steps=steps,
        learning_rate=2e-3 if terms == 1 else 1e-3,
        final_learning_rate=1e-5,
        seed=K1_SEED if terms == 1 else K4_GROWTH_SEED,
        device="cpu",
        record_points=10,
        checkpoint_every=steps,
        lbfgs_refinement_steps=lbfgs_steps,
        truth_maximum_dimension=300,
        particle_tensor_maximum_coefficients=(
            100_000 if terms == 1 else 1_100_000
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1-steps", type=int, default=80)
    parser.add_argument("--k4-steps", type=int, default=120)
    parser.add_argument("--k1-lbfgs-steps", type=int, default=40)
    parser.add_argument("--k4-lbfgs-steps", type=int, default=60)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase29_n6_soft_coulomb_pilot"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase29_n6_soft_coulomb_pilot.json"
        ),
    )
    args = parser.parse_args()
    if min(
        args.k1_steps,
        args.k4_steps,
        args.k1_lbfgs_steps,
        args.k4_lbfgs_steps,
    ) < 1:
        raise ValueError("all registered step counts must be positive")

    one_body = harmonic_pair_hamiltonian(
        DIMENSION, kappa=0.0, dtype=torch.complex128, device="cpu"
    )[0]
    interaction, factorization = soft_coulomb_operator(
        DIMENSION,
        quadrature_order=QUADRATURE,
        coupling=COUPLING,
        softening=SOFTENING,
        relative_threshold=1e-13,
        factorization_backend="physical",
        dtype=torch.complex128,
        device="cpu",
    )
    operators = (one_body, interaction)
    dense_truth = _dense_ci()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    k1_checkpoint = args.checkpoint_dir / "N6_D10_K1_seed29.pt"
    k1 = run_diagonal_path_variable_projection(
        _config(terms=1, steps=args.k1_steps, lbfgs_steps=args.k1_lbfgs_steps),
        checkpoint_path=k1_checkpoint,
        operators=operators,
        operator_id="soft_N6_D10_Q128_physical_svd_K1",
    )
    k1.update(
        {
            "point_id": "N6_D10_K1_seed29_blind",
            "initialization_lineage": {
                "kind": "blind_slater",
                "truth_state_used": False,
                "seed": K1_SEED,
            },
            "dense_quadrature_reference_energy": dense_truth["energy"],
            "error_vs_dense_quadrature_ci": k1["energy"] - dense_truth["energy"],
        }
    )
    print(k1["point_id"], k1["error_vs_dense_quadrature_ci"], flush=True)

    source = load_diagonal_path_checkpoint(k1_checkpoint)
    initial_k4 = extend_diagonal_path_terms(
        source["best_raw"], 4, seed=K4_GROWTH_SEED
    )
    k4 = run_diagonal_path_variable_projection(
        _config(terms=4, steps=args.k4_steps, lbfgs_steps=args.k4_lbfgs_steps),
        checkpoint_path=args.checkpoint_dir / "N6_D10_K4_seed2914_from_K1.pt",
        initial_orbitals=initial_k4,
        operators=operators,
        operator_id="soft_N6_D10_Q128_physical_svd_K4",
    )
    k4.update(
        {
            "point_id": "N6_D10_K4_seed2914_from_K1",
            "initialization_lineage": {
                "kind": "exact_K1_span_plus_three_seeded_blind_slaters",
                "source_checkpoint": str(k1_checkpoint),
                "truth_state_used": False,
                "new_term_seed": K4_GROWTH_SEED,
            },
            "dense_quadrature_reference_energy": dense_truth["energy"],
            "error_vs_dense_quadrature_ci": k4["energy"] - dense_truth["energy"],
        }
    )
    print(k4["point_id"], k4["error_vs_dense_quadrature_ci"], flush=True)

    operator_pass = bool(
        factorization.dense_relative_factorization_error <= FACTORIZATION_TOLERANCE
        and all(
            abs(point["finite_basis_reference_energy"] - dense_truth["energy"])
            <= FACTORIZATION_TOLERANCE
            for point in (k1, k4)
        )
    )
    k1_resource_pass = bool(
        k1["peak_cpu_rss_bytes"] <= MEMORY_LIMIT_BYTES
        and k1["total_elapsed_seconds_this_call"] <= TIME_LIMIT_SECONDS
    )
    k4_resource_pass = bool(
        k4["peak_cpu_rss_bytes"] <= MEMORY_LIMIT_BYTES
        and k4["total_elapsed_seconds_this_call"] <= TIME_LIMIT_SECONDS
    )
    k4_error = k4["error_vs_dense_quadrature_ci"]
    k1_error = k1["error_vs_dense_quadrature_ci"]
    nested_nonworsening = bool(
        k4["history"][0]["energy"] <= k1["energy"] + 1e-9
        and k4["energy"] <= k4["history"][0]["energy"] + 1e-9
    )
    state_pass = bool(
        k1["completed"]
        and k4["completed"]
        and k1["norm_error"] <= NORM_TOLERANCE
        and k4["norm_error"] <= NORM_TOLERANCE
        and k1["structural_antisymmetry_residual"] <= ANTISYMMETRY_TOLERANCE
        and k4["structural_antisymmetry_residual"] <= ANTISYMMETRY_TOLERANCE
        and k4["materialized_antisymmetry_residual"] <= ANTISYMMETRY_TOLERANCE
        and k1["structural_counts"]["enumerated_virtual_paths"] == 0
        and k4["structural_counts"]["enumerated_virtual_paths"] == 0
        and k4["structural_counts"]["materialized_particle_coefficients"] == 0
    )
    correlation_pass = bool(
        -1e-9 <= k4_error <= K4_ERROR_TOLERANCE
        and k4_error <= 0.5 * k1_error
        and k4["energy_variance"] <= K4_VARIANCE_TOLERANCE
        and nested_nonworsening
    )
    truth_pass = bool(
        dense_truth["norm_error"] <= NORM_TOLERANCE
        and dense_truth["antisymmetry_residual"] <= ANTISYMMETRY_TOLERANCE
        and dense_truth["energy_variance"] <= 1e-20
    )
    accepted = bool(
        operator_pass
        and k1_resource_pass
        and k4_resource_pass
        and state_pass
        and correlation_pass
        and truth_pass
    )
    artifact = {
        "schema_version": 1,
        "experiment": "phase29_resource_capped_N6_D10_soft_coulomb_pilot",
        "evidence_level": "numerical",
        "scientific_boundary": "single-seed feasibility pilot; not stability or scaling evidence",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "model": {
            "N": PARTICLES,
            "D": DIMENSION,
            "Q": QUADRATURE,
            "coupling": COUPLING,
            "softening": SOFTENING,
        },
        "thresholds": {
            "peak_cpu_rss_bytes": MEMORY_LIMIT_BYTES,
            "wall_time_seconds_per_point": TIME_LIMIT_SECONDS,
            "factorization_error": FACTORIZATION_TOLERANCE,
            "norm_error": NORM_TOLERANCE,
            "antisymmetry_residual": ANTISYMMETRY_TOLERANCE,
            "K4_dense_ci_error": K4_ERROR_TOLERANCE,
            "K4_error_ratio_vs_K1": 0.5,
            "K4_variance": K4_VARIANCE_TOLERANCE,
        },
        "operator_audit": {
            "backend": factorization.factorization_backend,
            "rank": factorization.retained_rank,
            "dense_relative_factorization_error": (
                factorization.dense_relative_factorization_error
            ),
        },
        "dense_ci_audit": dense_truth,
        "points": [k1, k4],
        "correlation_axis": [
            {"K": 1, "energy": k1["energy"], "error": k1_error},
            {"K": 4, "energy": k4["energy"], "error": k4_error},
        ],
        "diagnostics": {
            "K4_initial_nested_energy": k4["history"][0]["energy"],
            "nested_nonworsening": nested_nonworsening,
            "K4_error_ratio_vs_K1": k4_error / k1_error,
        },
        "acceptance": {
            "operator_pass": operator_pass,
            "truth_pass": truth_pass,
            "K1_resource_pass": k1_resource_pass,
            "K4_resource_pass": k4_resource_pass,
            "state_pass": state_pass,
            "correlation_pass": correlation_pass,
            "pilot_pass": accepted,
        },
        "config": {
            "K1_steps": args.k1_steps,
            "K4_steps": args.k4_steps,
            "K1_lbfgs_steps": args.k1_lbfgs_steps,
            "K4_lbfgs_steps": args.k4_lbfgs_steps,
        },
    }
    _write(args.output, artifact)
    print(json.dumps({"output": str(args.output), "pilot_pass": accepted}, indent=2))


if __name__ == "__main__":
    main()
