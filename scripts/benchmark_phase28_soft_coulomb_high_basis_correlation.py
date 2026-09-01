"""Audit blind K=4 -> 5 correlation growth at N=4,D=12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from femps.algorithms import (
    DiagonalPathConfig,
    extend_diagonal_path_terms,
    load_diagonal_path_checkpoint,
    run_diagonal_path_variable_projection,
)
from femps.hamiltonians import harmonic_pair_hamiltonian, soft_coulomb_operator

try:
    from scripts.benchmark_phase28_soft_coulomb_transferability import (
        ANTISYMMETRY_TOLERANCE,
        COUPLING,
        FACTORIZATION_TOLERANCE,
        FINITE_ERROR_TOLERANCE,
        NORM_TOLERANCE,
        PARTICLES,
        QUADRATURE,
        SOFTENING,
        VARIANCE_TOLERANCE,
        _dense_ci,
        _write,
    )
except ModuleNotFoundError:
    from benchmark_phase28_soft_coulomb_transferability import (
        ANTISYMMETRY_TOLERANCE,
        COUPLING,
        FACTORIZATION_TOLERANCE,
        FINITE_ERROR_TOLERANCE,
        NORM_TOLERANCE,
        PARTICLES,
        QUADRATURE,
        SOFTENING,
        VARIANCE_TOLERANCE,
        _dense_ci,
        _write,
    )


DIMENSION = 12
SOURCE_TERMS = 4
TARGET_TERMS = 5
SEED = 2812
MATERIAL_ERROR_REDUCTION = 0.10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lbfgs-steps", type=int, default=50)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/phase28_soft_coulomb_basis_extension/"
            "N4_D12_K4_seed17.pt"
        ),
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=Path(
            "docs/experiments/results/phase28_soft_coulomb_basis_extension.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/phase28_soft_coulomb_high_basis_correlation/"
            "N4_D12_K5_seed2812.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/"
            "phase28_soft_coulomb_high_basis_correlation.json"
        ),
    )
    args = parser.parse_args()
    if min(args.steps, args.lbfgs_steps) < 1:
        raise ValueError("step counts must be positive")
    if not args.source_checkpoint.exists():
        raise ValueError("missing D12,K4 source checkpoint; reproduce the basis extension")

    source_payload = load_diagonal_path_checkpoint(args.source_checkpoint)
    source_artifact = json.loads(args.source_artifact.read_text(encoding="utf-8"))
    source_point = source_artifact["extension_points"][1]
    if (
        source_point["config"]["basis_order"] != DIMENSION
        or source_point["config"]["terms"] != SOURCE_TERMS
    ):
        raise ValueError("source artifact does not contain the registered D12,K4 point")
    initial = extend_diagonal_path_terms(
        source_payload["best_raw"], TARGET_TERMS, seed=SEED
    )

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
    config = DiagonalPathConfig(
        basis_order=DIMENSION,
        particles=PARTICLES,
        terms=TARGET_TERMS,
        interaction_model="soft_coulomb",
        soft_coulomb_coupling=COUPLING,
        soft_coulomb_softening=SOFTENING,
        soft_coulomb_quadrature_order=QUADRATURE,
        steps=args.steps,
        learning_rate=8e-4,
        final_learning_rate=1e-5,
        seed=SEED,
        device="cpu",
        record_points=10,
        checkpoint_every=args.steps,
        lbfgs_refinement_steps=args.lbfgs_steps,
        truth_maximum_dimension=600,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    result = run_diagonal_path_variable_projection(
        config,
        checkpoint_path=args.checkpoint,
        initial_orbitals=initial,
        operators=(one_body, interaction),
        operator_id="soft_D12_Q128_physical_svd_K5_growth",
    )
    dense_truth = _dense_ci(DIMENSION)
    dense_error = result["energy"] - dense_truth["energy"]
    initial_energy = result["history"][0]["energy"]
    source_error = source_point["error_vs_dense_quadrature_ci"]
    result.update(
        {
            "point_id": "N4_D12_K5_seed2812_blind_term_growth",
            "initialization_lineage": {
                "kind": "exact_K4_span_plus_seeded_blind_slater",
                "source_checkpoint": str(args.source_checkpoint),
                "truth_state_used": False,
                "new_term_seed": SEED,
            },
            "dense_quadrature_reference_energy": dense_truth["energy"],
            "error_vs_dense_quadrature_ci": dense_error,
        }
    )

    source_nested = initial_energy <= source_point["energy"] + 1e-9
    optimized_nonworsening = result["energy"] <= initial_energy + 1e-9
    state_pass = bool(
        result["completed"]
        and -1e-9 <= dense_error <= FINITE_ERROR_TOLERANCE
        and result["energy_variance"] <= VARIANCE_TOLERANCE
        and result["norm_error"] <= NORM_TOLERANCE
        and result["structural_antisymmetry_residual"] <= ANTISYMMETRY_TOLERANCE
        and result["materialized_antisymmetry_residual"] <= ANTISYMMETRY_TOLERANCE
        and result["structural_counts"]["enumerated_virtual_paths"] == 0
        and result["peak_cpu_rss_bytes"] > 0
    )
    operator_pass = bool(
        diagnostics.dense_relative_factorization_error <= FACTORIZATION_TOLERANCE
        and abs(result["finite_basis_reference_energy"] - dense_truth["energy"])
        <= FACTORIZATION_TOLERANCE
    )
    audit_pass = source_nested and optimized_nonworsening and state_pass and operator_pass
    error_reduction_fraction = (source_error - dense_error) / source_error
    material_improvement = bool(
        error_reduction_fraction >= MATERIAL_ERROR_REDUCTION
        and result["energy_variance"] <= source_point["energy_variance"]
    )
    artifact = {
        "schema_version": 1,
        "experiment": "phase28_soft_coulomb_D12_K4_K5_correlation_growth",
        "evidence_level": "numerical",
        "scientific_boundary": "single seeded blind added determinant; not a K-scaling claim",
        "thresholds": {
            "dense_ci_error": FINITE_ERROR_TOLERANCE,
            "variance": VARIANCE_TOLERANCE,
            "norm_error": NORM_TOLERANCE,
            "antisymmetry_residual": ANTISYMMETRY_TOLERANCE,
            "factorization_error": FACTORIZATION_TOLERANCE,
            "material_error_reduction_fraction": MATERIAL_ERROR_REDUCTION,
        },
        "source_K4_point": source_point,
        "K5_point": result,
        "operator_audit": {
            "backend": diagnostics.factorization_backend,
            "rank": diagnostics.retained_rank,
            "dense_relative_factorization_error": (
                diagnostics.dense_relative_factorization_error
            ),
        },
        "dense_ci_audit": dense_truth,
        "correlation_axis": [
            {"K": SOURCE_TERMS, "energy": source_point["energy"], "error": source_error},
            {"K": TARGET_TERMS, "energy": result["energy"], "error": dense_error},
        ],
        "diagnostics": {
            "initial_nested_energy": initial_energy,
            "source_span_nonworsening": source_nested,
            "optimization_nonworsening": optimized_nonworsening,
            "error_reduction_fraction": error_reduction_fraction,
        },
        "acceptance": {
            "state_pass": state_pass,
            "operator_pass": operator_pass,
            "audit_pass": audit_pass,
            "material_improvement": material_improvement,
        },
        "config": {"steps": args.steps, "lbfgs_steps": args.lbfgs_steps},
    }
    _write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "audit_pass": audit_pass,
                "material_improvement": material_improvement,
                "error_reduction_fraction": error_reduction_fraction,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
