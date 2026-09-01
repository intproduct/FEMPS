"""Run the ADR-0019 blind N=6,D=10,K=4 stability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from femps.algorithms import DiagonalPathConfig, run_diagonal_path_variable_projection
from femps.hamiltonians import harmonic_pair_hamiltonian, soft_coulomb_operator


PARTICLES = 6
DIMENSION = 10
TERMS = 4
QUADRATURE = 128
COUPLING = 1.0
SOFTENING = 1.0
SEEDS = (31, 37, 43)
MEMORY_LIMIT_BYTES = 1_610_612_736
TIME_LIMIT_SECONDS = 600.0
FACTORIZATION_TOLERANCE = 1e-11
ERROR_TOLERANCE = 5e-4
VARIANCE_TOLERANCE = 5e-3
NORM_TOLERANCE = 1e-10
ANTISYMMETRY_TOLERANCE = 1e-12
CONDITION_LIMIT = 1e8
ENERGY_SPREAD_TOLERANCE = 2.5e-4


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--lbfgs-steps", type=int, default=80)
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=Path("docs/experiments/results/phase29_n6_soft_coulomb_pilot.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase29_n6_multiseed_stability"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase29_n6_multiseed_stability.json"
        ),
    )
    args = parser.parse_args()
    if min(args.steps, args.lbfgs_steps) < 1:
        raise ValueError("registered step counts must be positive")
    source = json.loads(args.source_artifact.read_text(encoding="utf-8"))
    truth = source["dense_ci_audit"]
    if truth["N"] != PARTICLES or truth["D"] != DIMENSION or truth["Q"] != QUADRATURE:
        raise ValueError("source truth does not match the ADR-0019 model")

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
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    points = []
    for seed in SEEDS:
        config = DiagonalPathConfig(
            basis_order=DIMENSION,
            particles=PARTICLES,
            terms=TERMS,
            interaction_model="soft_coulomb",
            soft_coulomb_coupling=COUPLING,
            soft_coulomb_softening=SOFTENING,
            soft_coulomb_quadrature_order=QUADRATURE,
            steps=args.steps,
            learning_rate=1e-3,
            final_learning_rate=1e-5,
            seed=seed,
            device="cpu",
            record_points=10,
            checkpoint_every=args.steps,
            lbfgs_refinement_steps=args.lbfgs_steps,
            truth_maximum_dimension=300,
            particle_tensor_maximum_coefficients=(
                1_100_000 if seed == SEEDS[0] else 100_000
            ),
        )
        result = run_diagonal_path_variable_projection(
            config,
            checkpoint_path=args.checkpoint_dir / f"N6_D10_K4_seed{seed}.pt",
            operators=(one_body, interaction),
            operator_id=f"soft_N6_D10_Q128_physical_svd_blind_seed{seed}",
        )
        result.update(
            {
                "point_id": f"N6_D10_K4_seed{seed}_blind",
                "initialization_lineage": {
                    "kind": "slater_plus_three_seeded_blind_slaters",
                    "truth_state_used": False,
                    "seed": seed,
                },
                "dense_quadrature_reference_energy": truth["energy"],
                "error_vs_dense_quadrature_ci": result["energy"] - truth["energy"],
            }
        )
        points.append(result)
        print(result["point_id"], result["error_vs_dense_quadrature_ci"], flush=True)

    point_pass = []
    for index, point in enumerate(points):
        materialization_pass = (
            point["materialized_antisymmetry_residual"] <= ANTISYMMETRY_TOLERANCE
            if index == 0
            else point["materialized_antisymmetry_residual"] is None
        )
        point_pass.append(
            bool(
                point["completed"]
                and -1e-9
                <= point["error_vs_dense_quadrature_ci"]
                <= ERROR_TOLERANCE
                and point["energy_variance"] <= VARIANCE_TOLERANCE
                and point["norm_error"] <= NORM_TOLERANCE
                and point["structural_antisymmetry_residual"]
                <= ANTISYMMETRY_TOLERANCE
                and materialization_pass
                and point["retained_rank"] == TERMS
                and point["retained_condition_number"] <= CONDITION_LIMIT
                and point["structural_counts"]["enumerated_virtual_paths"] == 0
                and point["structural_counts"]["materialized_particle_coefficients"]
                == 0
                and abs(point["finite_basis_reference_energy"] - truth["energy"])
                <= FACTORIZATION_TOLERANCE
                and point["peak_cpu_rss_bytes"] <= MEMORY_LIMIT_BYTES
                and point["total_elapsed_seconds_this_call"] <= TIME_LIMIT_SECONDS
            )
        )
    energy_spread = max(point["energy"] for point in points) - min(
        point["energy"] for point in points
    )
    operator_pass = bool(
        diagnostics.dense_relative_factorization_error <= FACTORIZATION_TOLERANCE
    )
    spread_pass = energy_spread <= ENERGY_SPREAD_TOLERANCE
    accepted = all(point_pass) and operator_pass and spread_pass
    artifact = {
        "schema_version": 1,
        "experiment": "phase29_N6_D10_K4_blind_multiseed_stability",
        "evidence_level": "numerical",
        "scientific_boundary": "three fixed blind seeds at one N,D,K point; not scaling evidence",
        "model": {
            "N": PARTICLES,
            "D": DIMENSION,
            "K": TERMS,
            "Q": QUADRATURE,
            "coupling": COUPLING,
            "softening": SOFTENING,
        },
        "seeds": list(SEEDS),
        "thresholds": {
            "dense_ci_error": ERROR_TOLERANCE,
            "variance": VARIANCE_TOLERANCE,
            "norm_error": NORM_TOLERANCE,
            "antisymmetry_residual": ANTISYMMETRY_TOLERANCE,
            "retained_condition_number": CONDITION_LIMIT,
            "factorization_error": FACTORIZATION_TOLERANCE,
            "peak_cpu_rss_bytes": MEMORY_LIMIT_BYTES,
            "wall_time_seconds_per_point": TIME_LIMIT_SECONDS,
            "energy_spread": ENERGY_SPREAD_TOLERANCE,
        },
        "operator_audit": {
            "backend": diagnostics.factorization_backend,
            "rank": diagnostics.retained_rank,
            "dense_relative_factorization_error": (
                diagnostics.dense_relative_factorization_error
            ),
        },
        "dense_ci_audit": truth,
        "points": points,
        "stability": {
            "runs": len(points),
            "passes": sum(point_pass),
            "per_run_pass": point_pass,
            "energy_spread": energy_spread,
            "maximum_dense_ci_error": max(
                point["error_vs_dense_quadrature_ci"] for point in points
            ),
            "maximum_variance": max(point["energy_variance"] for point in points),
            "maximum_condition_number": max(
                point["retained_condition_number"] for point in points
            ),
            "maximum_time_seconds": max(
                point["total_elapsed_seconds_this_call"] for point in points
            ),
            "maximum_peak_cpu_rss_bytes": max(
                point["peak_cpu_rss_bytes"] for point in points
            ),
        },
        "acceptance": {
            "operator_pass": operator_pass,
            "spread_pass": spread_pass,
            "multiseed_pass": accepted,
        },
        "config": {"steps": args.steps, "lbfgs_steps": args.lbfgs_steps},
    }
    _write(args.output, artifact)
    print(json.dumps({"output": str(args.output), "multiseed_pass": accepted}, indent=2))


if __name__ == "__main__":
    main()
