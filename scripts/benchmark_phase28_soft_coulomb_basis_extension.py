"""Run the registered D=8 -> 10 -> 12 soft-Coulomb FEMPS lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from femps.algorithms import (
    DiagonalPathConfig,
    embed_diagonal_path_orbitals,
    load_diagonal_path_checkpoint,
    run_diagonal_path_variable_projection,
)
from femps.hamiltonians import harmonic_pair_hamiltonian, soft_coulomb_operator
try:
    from scripts.benchmark_phase28_soft_coulomb_transferability import (
        ANTISYMMETRY_TOLERANCE,
        COUPLING,
        D14_REFERENCE,
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
        D14_REFERENCE,
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


def _run_extension(
    *,
    dimension: int,
    learning_rate: float,
    source_checkpoint: Path,
    output_checkpoint: Path,
    steps: int,
    lbfgs_steps: int,
) -> tuple[dict, dict]:
    source = load_diagonal_path_checkpoint(source_checkpoint)
    initial = embed_diagonal_path_orbitals(source["best_raw"], dimension)
    one_body = harmonic_pair_hamiltonian(
        dimension, kappa=0.0, dtype=torch.complex128, device="cpu"
    )[0]
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
    config = DiagonalPathConfig(
        basis_order=dimension,
        particles=PARTICLES,
        terms=4,
        interaction_model="soft_coulomb",
        soft_coulomb_coupling=COUPLING,
        soft_coulomb_softening=SOFTENING,
        soft_coulomb_quadrature_order=QUADRATURE,
        steps=steps,
        learning_rate=learning_rate,
        final_learning_rate=1e-5,
        seed=17,
        device="cpu",
        record_points=10,
        checkpoint_every=steps,
        lbfgs_refinement_steps=lbfgs_steps,
        truth_maximum_dimension=600,
    )
    result = run_diagonal_path_variable_projection(
        config,
        checkpoint_path=output_checkpoint,
        initial_orbitals=initial,
        operators=(one_body, interaction),
        operator_id=f"soft_D{dimension}_Q{QUADRATURE}_physical_svd",
    )
    dense_truth = _dense_ci(dimension)
    result.update(
        {
            "point_id": f"N4_D{dimension}_K4_seed17_basis_extension",
            "initialization_lineage": {
                "kind": "nested_basis_zero_padding",
                "source_checkpoint": str(source_checkpoint),
                "truth_state_used": False,
            },
            "dense_quadrature_reference_energy": dense_truth["energy"],
            "error_vs_dense_quadrature_ci": result["energy"] - dense_truth["energy"],
            "error_vs_D14_numerical_reference": result["energy"] - D14_REFERENCE,
        }
    )
    operator_record = {
        "D": dimension,
        "Q": QUADRATURE,
        "backend": diagnostics.factorization_backend,
        "rank": diagnostics.retained_rank,
        "relative_spectral_threshold": diagnostics.relative_threshold,
        "dense_relative_factorization_error": diagnostics.dense_relative_factorization_error,
    }
    return result, {"operator": operator_record, "dense_ci": dense_truth}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lbfgs-steps", type=int, default=40)
    parser.add_argument(
        "--source-d8-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/phase28_soft_coulomb_transferability/"
            "N4_D8_K4_seed17_from_D6.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase28_soft_coulomb_basis_extension"),
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=Path(
            "docs/experiments/results/phase28_soft_coulomb_transferability.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase28_soft_coulomb_basis_extension.json"
        ),
    )
    args = parser.parse_args()
    if min(args.steps, args.lbfgs_steps) < 1:
        raise ValueError("step counts must be positive")
    if not args.source_d8_checkpoint.exists():
        raise ValueError(
            "missing D8 source checkpoint; reproduce the transferability benchmark first"
        )
    source_artifact = json.loads(args.source_artifact.read_text(encoding="utf-8"))
    d8 = source_artifact["d8_nested_basis_continuation_multiseed"][0]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    d10, audit10 = _run_extension(
        dimension=10,
        learning_rate=2e-3,
        source_checkpoint=args.source_d8_checkpoint,
        output_checkpoint=args.checkpoint_dir / "N4_D10_K4_seed17.pt",
        steps=args.steps,
        lbfgs_steps=args.lbfgs_steps,
    )
    print(d10["point_id"], d10["error_vs_dense_quadrature_ci"], flush=True)
    d12, audit12 = _run_extension(
        dimension=12,
        learning_rate=1e-3,
        source_checkpoint=args.checkpoint_dir / "N4_D10_K4_seed17.pt",
        output_checkpoint=args.checkpoint_dir / "N4_D12_K4_seed17.pt",
        steps=args.steps,
        lbfgs_steps=args.lbfgs_steps,
    )
    print(d12["point_id"], d12["error_vs_dense_quadrature_ci"], flush=True)

    points = [d10, d12]
    point_pass = [
        bool(
            p["completed"]
            and -1e-9 <= p["error_vs_dense_quadrature_ci"] <= FINITE_ERROR_TOLERANCE
            and p["energy_variance"] <= VARIANCE_TOLERANCE
            and p["norm_error"] <= NORM_TOLERANCE
            and p["structural_antisymmetry_residual"] <= ANTISYMMETRY_TOLERANCE
            and p["materialized_antisymmetry_residual"] <= ANTISYMMETRY_TOLERANCE
            and p["structural_counts"]["enumerated_virtual_paths"] == 0
            and p["peak_cpu_rss_bytes"] > 0
        )
        for p in points
    ]
    operator_pass = all(
        audit["operator"]["dense_relative_factorization_error"]
        <= FACTORIZATION_TOLERANCE
        for audit in (audit10, audit12)
    )
    d_axis = [
        {
            "D": 8,
            "energy": d8["energy"],
            "absolute_error_vs_D14": abs(d8["error_vs_D14_numerical_reference"]),
        },
        *[
            {
                "D": p["config"]["basis_order"],
                "energy": p["energy"],
                "absolute_error_vs_D14": abs(p["error_vs_D14_numerical_reference"]),
            }
            for p in points
        ],
    ]
    monotone = all(
        b["absolute_error_vs_D14"] <= a["absolute_error_vs_D14"] + 1e-9
        for a, b in zip(d_axis, d_axis[1:])
    )
    accepted = all(point_pass) and operator_pass and monotone
    artifact = {
        "schema_version": 1,
        "experiment": "phase28_soft_coulomb_D8_D10_D12_basis_extension",
        "evidence_level": "numerical",
        "scientific_boundary": "single seed bounded basis lineage; not a scaling claim",
        "D14_reference": {
            "energy": D14_REFERENCE,
            "role": "finite-basis numerical reference, not a continuum bound",
        },
        "thresholds": {
            "dense_ci_error": FINITE_ERROR_TOLERANCE,
            "variance": VARIANCE_TOLERANCE,
            "norm_error": NORM_TOLERANCE,
            "antisymmetry_residual": ANTISYMMETRY_TOLERANCE,
            "factorization_error": FACTORIZATION_TOLERANCE,
        },
        "source_D8_point": d8,
        "extension_points": points,
        "operator_and_truth_audits": [audit10, audit12],
        "convergence": {
            "D_axis_K4": d_axis,
            "absolute_error_vs_D14_nonincreasing": monotone,
        },
        "acceptance": {
            "per_point_pass": point_pass,
            "operator_factorization_pass": operator_pass,
            "basis_extension_pass": accepted,
        },
        "config": {"steps": args.steps, "lbfgs_steps": args.lbfgs_steps},
    }
    _write(args.output, artifact)
    print(json.dumps({"output": str(args.output), "basis_extension_pass": accepted}, indent=2))


if __name__ == "__main__":
    main()
