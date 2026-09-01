"""Close the Phase 28 interacting N=4 FEMPS benchmark gate."""

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
    FiniteAgpConfig,
    embed_diagonal_path_orbitals,
    run_diagonal_path_variable_projection,
    run_finite_agp_variable_projection,
)
from femps.benchmarks import ProcessRSSMonitor
from femps.exterior import (
    antisymmetry_residual,
    exterior_coefficients_to_tensor,
    particle_tt_ranks,
)
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian,
    exact_interacting_harmonic_fermion_energy,
    harmonic_pair_hamiltonian,
)


PARTICLES = 4
KAPPA = 0.35
STABILITY_FINITE_ERROR = 1e-3
STABILITY_VARIANCE = 1e-2
ANTISYMMETRY_TOLERANCE = 1e-12


def _point_id(dimension: int, terms: int, seed: int, lineage: str) -> str:
    return f"N4_D{dimension}_K{terms}_seed{seed}_{lineage}"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_point(
    *,
    dimension: int,
    terms: int,
    seed: int,
    steps: int,
    learning_rate: float,
    lbfgs_steps: int,
    checkpoint_dir: Path,
    lineage: str,
    source_checkpoint: Path | None = None,
) -> tuple[dict, Path]:
    point_id = _point_id(dimension, terms, seed, lineage)
    checkpoint = checkpoint_dir / f"{point_id}.pt"
    config = DiagonalPathConfig(
        basis_order=dimension,
        particles=PARTICLES,
        terms=terms,
        kappa=KAPPA,
        steps=steps,
        learning_rate=learning_rate,
        final_learning_rate=1e-5,
        seed=seed,
        device="cpu",
        record_points=12,
        checkpoint_every=steps,
        lbfgs_refinement_steps=lbfgs_steps,
    )
    initial_orbitals = None
    if source_checkpoint is not None:
        source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        initial_orbitals = embed_diagonal_path_orbitals(
            source["best_raw"], dimension
        )
    result = run_diagonal_path_variable_projection(
        config,
        checkpoint_path=checkpoint,
        initial_orbitals=initial_orbitals,
    )
    result["point_id"] = point_id
    result["initialization_lineage"] = (
        {
            "kind": "nested_basis_continuation",
            "source_checkpoint": str(source_checkpoint),
            "truth_state_used": False,
        }
        if source_checkpoint is not None
        else {
            "kind": "blind_slater_plus_seeded_random",
            "source_checkpoint": None,
            "truth_state_used": False,
        }
    )
    return result, checkpoint


def _exact_ci_comparator(dimension: int) -> dict:
    started = time.perf_counter()
    with ProcessRSSMonitor() as monitor:
        one_body, interaction = harmonic_pair_hamiltonian(
            dimension, kappa=KAPPA, dtype=torch.complex128, device="cpu"
        )
        hamiltonian = antisymmetric_many_body_hamiltonian(
            one_body, PARTICLES, interaction
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
        coefficients = eigenvectors[:, 0]
        energy = eigenvalues[0].real
        residual = hamiltonian @ coefficients - energy * coefficients
        variance = torch.vdot(residual, residual).real
        particle_state = exterior_coefficients_to_tensor(
            coefficients, dimension, PARTICLES
        )
        ranks = particle_tt_ranks(particle_state)
        symmetry_residual = antisymmetry_residual(particle_state)
    memory = monitor.record()
    exterior_dimension = math.comb(dimension, PARTICLES)
    continuum = exact_interacting_harmonic_fermion_energy(
        PARTICLES, kappa=KAPPA
    )
    return {
        "method": "exact_diagonalization_finite_basis_ci",
        "role": "bounded_truth_comparator_not_femps_production",
        "basis_order": dimension,
        "particles": PARTICLES,
        "energy": float(energy),
        "continuum_reference_energy": continuum,
        "error_vs_continuum": float(energy) - continuum,
        "energy_variance": float(variance),
        "norm_error": float(abs(torch.vdot(coefficients, coefficients).real - 1.0)),
        "antisymmetry_residual": float(symmetry_residual),
        "exterior_coefficients": exterior_dimension,
        "dense_hamiltonian_entries": exterior_dimension**2,
        "materialized_particle_coefficients": dimension**PARTICLES,
        "ordinary_particle_tt_ranks": list(ranks),
        "elapsed_seconds": time.perf_counter() - started,
        "cpu_memory": memory.as_dict(),
    }


def _stability_summary(points: list[dict]) -> dict:
    passed = []
    for point in points:
        success = bool(
            point["completed"]
            and point["error_vs_finite_basis"] >= -1e-9
            and point["error_vs_finite_basis"] <= STABILITY_FINITE_ERROR
            and point["energy_variance"] <= STABILITY_VARIANCE
            and point["structural_antisymmetry_residual"]
            <= ANTISYMMETRY_TOLERANCE
            and (
                point["materialized_antisymmetry_residual"] is None
                or point["materialized_antisymmetry_residual"]
                <= ANTISYMMETRY_TOLERANCE
            )
            and point["structural_counts"]["enumerated_virtual_paths"] == 0
        )
        passed.append(success)
    energies = [point["energy"] for point in points]
    return {
        "runs": len(points),
        "passes": sum(passed),
        "success_rate": sum(passed) / len(points),
        "all_pass": all(passed),
        "energy_spread": max(energies) - min(energies),
        "maximum_finite_basis_error": max(
            point["error_vs_finite_basis"] for point in points
        ),
        "maximum_energy_variance": max(
            point["energy_variance"] for point in points
        ),
        "criteria": {
            "finite_basis_error_at_most": STABILITY_FINITE_ERROR,
            "energy_variance_at_most": STABILITY_VARIANCE,
            "antisymmetry_residual_at_most": ANTISYMMETRY_TOLERANCE,
        },
        "per_run_pass": passed,
    }


def _agp_comparator(steps: int) -> dict:
    config = FiniteAgpConfig(
        basis_order=6,
        particles=PARTICLES,
        agp_terms=1,
        kappa=KAPPA,
        steps=steps,
        learning_rate=5e-3,
        final_learning_rate=1e-5,
        seed=0,
        device="cpu",
        record_points=12,
        checkpoint_every=steps,
    )
    started = time.perf_counter()
    with ProcessRSSMonitor() as monitor:
        result = run_finite_agp_variable_projection(config)
    memory = monitor.record()
    return {
        "method": "finite_single_agp_comparator",
        "role": "first_quantized_pfaffian_agp_comparator_not_femps",
        "config": asdict(config),
        "energy": result["final_energy"],
        "finite_basis_reference_energy": result["finite_basis_reference_energy"],
        "continuum_reference_energy": result["continuum_reference_energy"],
        "error_vs_finite_basis": result["error_vs_finite_basis"],
        "error_vs_continuum": result["error_vs_continuum"],
        "energy_variance": result["energy_variance"],
        "norm_error": result["norm_error"],
        "antisymmetry_residual": result["antisymmetry_residual"],
        "ordinary_particle_tt_ranks": result["ordinary_particle_tt_ranks"],
        "elapsed_seconds": time.perf_counter() - started,
        "cpu_memory": memory.as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 41])
    parser.add_argument("--d6-steps", type=int, default=140)
    parser.add_argument("--d7-steps", type=int, default=260)
    parser.add_argument("--lbfgs-steps", type=int, default=80)
    parser.add_argument("--agp-steps", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/phase28_e4_closure.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase28_e4_closure"),
    )
    args = parser.parse_args()
    if not args.seeds or min(
        args.d6_steps, args.d7_steps, args.lbfgs_steps, args.agp_steps
    ) < 1:
        raise ValueError("seeds and all iteration counts must be positive")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    d6_blind = []
    d6_checkpoints: dict[int, Path] = {}
    for seed in args.seeds:
        point, checkpoint = _run_point(
            dimension=6,
            terms=4,
            seed=seed,
            steps=args.d6_steps,
            learning_rate=5e-3,
            lbfgs_steps=args.lbfgs_steps,
            checkpoint_dir=args.checkpoint_dir,
            lineage="blind",
        )
        d6_blind.append(point)
        d6_checkpoints[seed] = checkpoint
        print(
            f"D6 K4 seed={seed}: finite_error={point['error_vs_finite_basis']:.3e} "
            f"variance={point['energy_variance']:.3e}",
            flush=True,
        )

    d7_continuation = []
    for seed in args.seeds:
        point, _ = _run_point(
            dimension=7,
            terms=4,
            seed=seed,
            steps=args.d7_steps,
            learning_rate=2e-3,
            lbfgs_steps=args.lbfgs_steps,
            checkpoint_dir=args.checkpoint_dir,
            lineage="from_D6",
            source_checkpoint=d6_checkpoints[seed],
        )
        d7_continuation.append(point)
        print(
            f"D7 K4 seed={seed}: finite_error={point['error_vs_finite_basis']:.3e} "
            f"variance={point['energy_variance']:.3e}",
            flush=True,
        )

    axis_points = []
    for dimension, terms, steps in [(6, 1, 80), (6, 2, 120), (5, 4, 100)]:
        point, _ = _run_point(
            dimension=dimension,
            terms=terms,
            seed=args.seeds[0],
            steps=steps,
            learning_rate=5e-3,
            lbfgs_steps=args.lbfgs_steps,
            checkpoint_dir=args.checkpoint_dir,
            lineage="axis",
        )
        axis_points.append(point)

    k_axis = [
        next(point for point in axis_points if point["config"]["terms"] == terms)
        if terms < 4
        else d6_blind[0]
        for terms in (1, 2, 4)
    ]
    d_axis = [
        next(point for point in axis_points if point["config"]["basis_order"] == 5),
        d6_blind[0],
        d7_continuation[0],
    ]
    k_monotone = all(
        right["energy"] <= left["energy"] + 1e-9
        for left, right in zip(k_axis, k_axis[1:])
    )
    d_errors = [abs(point["error_vs_continuum"]) for point in d_axis]
    d_monotone = all(
        right <= left + 1e-9
        for left, right in zip(d_errors, d_errors[1:])
    )

    exact_ci_d6 = _exact_ci_comparator(6)
    exact_ci_d7 = _exact_ci_comparator(7)
    agp = _agp_comparator(args.agp_steps)
    d6_stability = _stability_summary(d6_blind)
    d7_stability = _stability_summary(d7_continuation)
    all_points = d6_blind + d7_continuation + axis_points
    cpu_memory_complete = all(
        point["peak_cpu_rss_bytes"] > 0
        and point["cpu_memory"]["samples"] >= 2
        for point in all_points
    )
    e4_pass = bool(
        d6_stability["all_pass"]
        and d7_stability["all_pass"]
        and k_monotone
        and d_monotone
        and cpu_memory_complete
    )
    artifact = {
        "schema_version": 1,
        "experiment": "phase28_e4_diagonal_path_closure",
        "evidence_level": "numerical",
        "state_definition": (
            "first-quantized continuous functional-basis diagonal-path "
            "matrix-wedge FEMPS"
        ),
        "scientific_boundary": (
            "restricted nonorthogonal multideterminant FEMPS; not generic "
            "FEMPS and not a second-quantized MPS"
        ),
        "seeds": args.seeds,
        "d6_blind_multiseed": d6_blind,
        "d7_nested_basis_continuation_multiseed": d7_continuation,
        "axis_points": axis_points,
        "stability": {"D6_blind": d6_stability, "D7_continuation": d7_stability},
        "convergence": {
            "K_axis_D6": [
                {"K": point["config"]["terms"], "energy": point["energy"]}
                for point in k_axis
            ],
            "K_axis_energy_nonincreasing": k_monotone,
            "D_axis_K4": [
                {
                    "D": point["config"]["basis_order"],
                    "energy": point["energy"],
                    "absolute_continuum_error": abs(point["error_vs_continuum"]),
                }
                for point in d_axis
            ],
            "D_axis_continuum_error_nonincreasing": d_monotone,
        },
        "comparators": {
            "single_slater_D6": k_axis[0],
            "exact_ci_D6": exact_ci_d6,
            "exact_ci_D7": exact_ci_d7,
            "single_agp_D6": agp,
            "ordinary_particle_tt": {
                "role": "exact materialized rank/resource comparator",
                "D6_exact_ci_ranks": exact_ci_d6["ordinary_particle_tt_ranks"],
                "D7_exact_ci_ranks": exact_ci_d7["ordinary_particle_tt_ranks"],
                "D6_materialized_coefficients": 6**PARTICLES,
                "D7_materialized_coefficients": 7**PARTICLES,
            },
        },
        "acceptance": {
            "E4_pass": e4_pass,
            "cpu_peak_memory_complete": cpu_memory_complete,
            "no_virtual_path_enumeration": all(
                point["structural_counts"]["enumerated_virtual_paths"] == 0
                for point in all_points
            ),
            "all_structural_antisymmetry_residuals_within_tolerance": all(
                point["structural_antisymmetry_residual"]
                <= ANTISYMMETRY_TOLERANCE
                for point in all_points
            ),
        },
    }
    _write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "E4_pass": e4_pass,
                "D6_success_rate": d6_stability["success_rate"],
                "D7_success_rate": d7_stability["success_rate"],
                "K_monotone": k_monotone,
                "D_monotone": d_monotone,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
