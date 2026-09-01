"""Run the registered nonquadratic soft-Coulomb FEMPS transfer test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from femps.algorithms import (
    DiagonalPathConfig,
    embed_diagonal_path_orbitals,
    load_diagonal_path_checkpoint,
    run_diagonal_path_variable_projection,
)
from femps.benchmarks import ProcessRSSMonitor
from femps.exterior import antisymmetry_residual, exterior_coefficients_to_tensor, particle_tt_ranks
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)


PARTICLES = 4
COUPLING = 1.0
SOFTENING = 1.0
QUADRATURE = 128
D14_REFERENCE = 11.023082853674637
ANTISYMMETRY_TOLERANCE = 1e-12
NORM_TOLERANCE = 1e-10
FINITE_ERROR_TOLERANCE = 2e-4
VARIANCE_TOLERANCE = 2e-3
FACTORIZATION_TOLERANCE = 1e-11


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _operators(dimension: int):
    one_body = harmonic_pair_hamiltonian(
        dimension, kappa=0.0, dtype=torch.complex128, device="cpu"
    )[0]
    interaction = None
    diagnostics = None
    for relative_threshold in (1e-13, 1e-14, 0.0):
        interaction, diagnostics = soft_coulomb_operator(
            dimension,
            quadrature_order=QUADRATURE,
            coupling=COUPLING,
            softening=SOFTENING,
            relative_threshold=relative_threshold,
            dtype=torch.complex128,
            device="cpu",
        )
        if diagnostics.dense_relative_factorization_error <= FACTORIZATION_TOLERANCE:
            break
    assert interaction is not None and diagnostics is not None
    if diagnostics.dense_relative_factorization_error > FACTORIZATION_TOLERANCE:
        raise RuntimeError("soft-Coulomb factorization cannot meet the registered tolerance")
    return (one_body, interaction), {
        "D": dimension,
        "Q": QUADRATURE,
        "factor_rank": diagnostics.retained_rank,
        "relative_spectral_threshold": diagnostics.relative_threshold,
        "dense_relative_factorization_error": diagnostics.dense_relative_factorization_error,
    }


def _dense_ci(dimension: int) -> dict:
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
        hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
            one_body, PARTICLES, dense_pair
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
        energy = eigenvalues[0].real
        coefficients = eigenvectors[:, 0]
        residual = hamiltonian @ coefficients - energy * coefficients
        particle_state = exterior_coefficients_to_tensor(
            coefficients, dimension, PARTICLES
        )
    memory = monitor.record()
    return {
        "method": "dense_quadrature_finite_basis_ci",
        "role": "bounded independent truth, not FEMPS production",
        "D": dimension,
        "Q": QUADRATURE,
        "energy": float(energy),
        "error_vs_D14_numerical_reference": float(energy) - D14_REFERENCE,
        "energy_variance": float(torch.vdot(residual, residual).real),
        "norm_error": float(abs(torch.vdot(coefficients, coefficients).real - 1.0)),
        "antisymmetry_residual": float(antisymmetry_residual(particle_state)),
        "ordinary_particle_tt_ranks": list(particle_tt_ranks(particle_state)),
        "exterior_dimension": math.comb(dimension, PARTICLES),
        "elapsed_seconds": time.perf_counter() - started,
        "cpu_memory": memory.as_dict(),
    }


def _run_point(
    *,
    dimension: int,
    terms: int,
    seed: int,
    steps: int,
    lbfgs_steps: int,
    checkpoint_dir: Path,
    operators,
    dense_truth: dict,
    source_checkpoint: Path | None = None,
) -> tuple[dict, Path]:
    lineage = "blind" if source_checkpoint is None else "from_D6"
    point_id = f"N4_D{dimension}_K{terms}_seed{seed}_{lineage}"
    checkpoint = checkpoint_dir / f"{point_id}.pt"
    config = DiagonalPathConfig(
        basis_order=dimension,
        particles=PARTICLES,
        terms=terms,
        interaction_model="soft_coulomb",
        soft_coulomb_coupling=COUPLING,
        soft_coulomb_softening=SOFTENING,
        soft_coulomb_quadrature_order=QUADRATURE,
        steps=steps,
        learning_rate=5e-3 if dimension == 6 else 2e-3,
        final_learning_rate=1e-5,
        seed=seed,
        device="cpu",
        record_points=10,
        checkpoint_every=steps,
        lbfgs_refinement_steps=lbfgs_steps,
    )
    initial = None
    if source_checkpoint is not None:
        payload = load_diagonal_path_checkpoint(source_checkpoint)
        initial = embed_diagonal_path_orbitals(payload["best_raw"], dimension)
    result = run_diagonal_path_variable_projection(
        config,
        checkpoint_path=checkpoint,
        initial_orbitals=initial,
        operators=operators,
        operator_id=f"soft_coulomb_D{dimension}_Q{QUADRATURE}",
    )
    result.update(
        {
            "point_id": point_id,
            "initialization_lineage": {
                "kind": "blind_slater_plus_seeded_random" if initial is None else "nested_basis_continuation",
                "source_checkpoint": None if source_checkpoint is None else str(source_checkpoint),
                "truth_state_used": False,
            },
            "dense_quadrature_reference_energy": dense_truth["energy"],
            "error_vs_dense_quadrature_ci": result["energy"] - dense_truth["energy"],
            "error_vs_D14_numerical_reference": result["energy"] - D14_REFERENCE,
        }
    )
    return result, checkpoint


def _stability(points: list[dict]) -> dict:
    per_run = []
    for point in points:
        per_run.append(
            bool(
                point["completed"]
                and -1e-9 <= point["error_vs_dense_quadrature_ci"] <= FINITE_ERROR_TOLERANCE
                and point["energy_variance"] <= VARIANCE_TOLERANCE
                and point["norm_error"] <= NORM_TOLERANCE
                and point["structural_antisymmetry_residual"] <= ANTISYMMETRY_TOLERANCE
                and point["materialized_antisymmetry_residual"] <= ANTISYMMETRY_TOLERANCE
                and point["structural_counts"]["enumerated_virtual_paths"] == 0
                and point["peak_cpu_rss_bytes"] > 0
            )
        )
    return {
        "runs": len(points),
        "passes": sum(per_run),
        "all_pass": all(per_run),
        "per_run_pass": per_run,
        "maximum_dense_ci_error": max(p["error_vs_dense_quadrature_ci"] for p in points),
        "maximum_variance": max(p["energy_variance"] for p in points),
        "energy_spread": max(p["energy"] for p in points) - min(p["energy"] for p in points),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 41])
    parser.add_argument("--d6-steps", type=int, default=60)
    parser.add_argument("--d8-steps", type=int, default=80)
    parser.add_argument("--lbfgs-steps", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/phase28_soft_coulomb_transferability.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase28_soft_coulomb_transferability"),
    )
    args = parser.parse_args()
    if not args.seeds or min(args.d6_steps, args.d8_steps, args.lbfgs_steps) < 1:
        raise ValueError("seeds and step counts must be positive")

    operator_data = {}
    operator_diagnostics = {}
    dense_truth = {}
    for dimension in (6, 8):
        operator_data[dimension], operator_diagnostics[dimension] = _operators(dimension)
        dense_truth[dimension] = _dense_ci(dimension)

    d6_points = []
    checkpoints = {}
    for seed in args.seeds:
        point, checkpoint = _run_point(
            dimension=6,
            terms=4,
            seed=seed,
            steps=args.d6_steps,
            lbfgs_steps=args.lbfgs_steps,
            checkpoint_dir=args.checkpoint_dir,
            operators=operator_data[6],
            dense_truth=dense_truth[6],
        )
        d6_points.append(point)
        checkpoints[seed] = checkpoint
        print(point["point_id"], point["error_vs_dense_quadrature_ci"], flush=True)

    d8_points = []
    for seed in args.seeds:
        point, _ = _run_point(
            dimension=8,
            terms=4,
            seed=seed,
            steps=args.d8_steps,
            lbfgs_steps=args.lbfgs_steps,
            checkpoint_dir=args.checkpoint_dir,
            operators=operator_data[8],
            dense_truth=dense_truth[8],
            source_checkpoint=checkpoints[seed],
        )
        d8_points.append(point)
        print(point["point_id"], point["error_vs_dense_quadrature_ci"], flush=True)

    k_points = []
    for terms in (1, 2):
        point, _ = _run_point(
            dimension=6,
            terms=terms,
            seed=args.seeds[0],
            steps=args.d6_steps,
            lbfgs_steps=args.lbfgs_steps,
            checkpoint_dir=args.checkpoint_dir,
            operators=operator_data[6],
            dense_truth=dense_truth[6],
        )
        k_points.append(point)
    k_axis = k_points + [d6_points[0]]
    k_monotone = all(b["energy"] <= a["energy"] + 1e-9 for a, b in zip(k_axis, k_axis[1:]))
    d_axis_errors = [
        abs(d6_points[0]["error_vs_D14_numerical_reference"]),
        abs(d8_points[0]["error_vs_D14_numerical_reference"]),
    ]
    d_monotone = d_axis_errors[1] <= d_axis_errors[0] + 1e-9
    d6_stability = _stability(d6_points)
    d8_stability = _stability(d8_points)
    factorization_pass = all(
        d["dense_relative_factorization_error"] <= FACTORIZATION_TOLERANCE
        for d in operator_diagnostics.values()
    )
    accepted = bool(
        d6_stability["all_pass"]
        and d8_stability["all_pass"]
        and k_monotone
        and d_monotone
        and factorization_pass
    )
    artifact = {
        "schema_version": 1,
        "experiment": "phase28_soft_coulomb_diagonal_path_transferability",
        "evidence_level": "numerical",
        "state_definition": "first-quantized continuous functional-basis diagonal-path matrix-wedge FEMPS",
        "scientific_boundary": "restricted nonorthogonal multideterminant FEMPS; not generic FEMPS or second quantization",
        "hamiltonian": "spin-polarized N=4 harmonic trap plus g/sqrt((x_i-x_j)^2+a^2), g=a=omega=1",
        "D14_reference": {
            "energy": D14_REFERENCE,
            "role": "pre-existing dense finite-basis numerical reference, not a continuum bound",
        },
        "thresholds": {
            "dense_ci_error": FINITE_ERROR_TOLERANCE,
            "variance": VARIANCE_TOLERANCE,
            "norm_error": NORM_TOLERANCE,
            "antisymmetry_residual": ANTISYMMETRY_TOLERANCE,
            "factorization_error": FACTORIZATION_TOLERANCE,
        },
        "operator_diagnostics": [operator_diagnostics[d] for d in (6, 8)],
        "dense_ci_comparators": [dense_truth[d] for d in (6, 8)],
        "d6_blind_multiseed": d6_points,
        "d8_nested_basis_continuation_multiseed": d8_points,
        "axis_points": k_points,
        "stability": {"D6_blind": d6_stability, "D8_continuation": d8_stability},
        "convergence": {
            "K_axis_D6": [{"K": p["config"]["terms"], "energy": p["energy"]} for p in k_axis],
            "K_axis_energy_nonincreasing": k_monotone,
            "D_axis_K4": [
                {"D": p["config"]["basis_order"], "energy": p["energy"], "absolute_error_vs_D14": abs(p["error_vs_D14_numerical_reference"])}
                for p in (d6_points[0], d8_points[0])
            ],
            "D_axis_error_vs_D14_nonincreasing": d_monotone,
        },
        "acceptance": {
            "transferability_pass": accepted,
            "operator_factorization_pass": factorization_pass,
            "no_virtual_path_enumeration": all(
                p["structural_counts"]["enumerated_virtual_paths"] == 0
                for p in d6_points + d8_points + k_points
            ),
        },
        "config": {"seeds": args.seeds, "d6_steps": args.d6_steps, "d8_steps": args.d8_steps, "lbfgs_steps": args.lbfgs_steps},
    }
    _write(args.output, artifact)
    print(json.dumps({"output": str(args.output), "transferability_pass": accepted}, indent=2))


if __name__ == "__main__":
    main()
