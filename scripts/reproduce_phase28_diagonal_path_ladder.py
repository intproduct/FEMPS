"""Reproduce the accepted portion of the Phase 28 FEMPS physics ladder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from femps.algorithms import (
    DiagonalPathConfig,
    run_diagonal_path_variable_projection,
)


def _point_id(*, particles: int, basis_order: int, terms: int, kappa: float) -> str:
    return f"N{particles}_D{basis_order}_K{terms}_kappa{kappa:g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kappa", type=float, default=0.35)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse matching completed points already present in --output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase28_diagonal_path_ladder.json"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase28_diagonal_path_ladder"),
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    specifications = [
        {"label": "E1", "particles": 2, "basis_order": 6, "terms": 1, "kappa": 0.0},
        {"label": "E2_D", "particles": 2, "basis_order": 4, "terms": 4, "kappa": args.kappa},
        {"label": "E2_K", "particles": 2, "basis_order": 6, "terms": 1, "kappa": args.kappa},
        {"label": "E2_K", "particles": 2, "basis_order": 6, "terms": 2, "kappa": args.kappa},
        {"label": "E2_DK", "particles": 2, "basis_order": 6, "terms": 4, "kappa": args.kappa},
        {"label": "E2_D", "particles": 2, "basis_order": 8, "terms": 4, "kappa": args.kappa},
        {"label": "E3", "particles": 4, "basis_order": 6, "terms": 1, "kappa": 0.0},
        {"label": "E4_K", "particles": 4, "basis_order": 6, "terms": 1, "kappa": args.kappa},
        {"label": "E4_K", "particles": 4, "basis_order": 6, "terms": 2, "kappa": args.kappa},
        {"label": "E4_K", "particles": 4, "basis_order": 6, "terms": 4, "kappa": args.kappa},
        {"label": "E4_D", "particles": 4, "basis_order": 5, "terms": 4, "kappa": args.kappa},
        {"label": "E4_D", "particles": 4, "basis_order": 7, "terms": 4, "kappa": args.kappa},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.reuse_existing and args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        existing = {point["point_id"]: point for point in prior.get("points", [])}
    points = []
    for specification in specifications:
        point_id = _point_id(
            particles=specification["particles"],
            basis_order=specification["basis_order"],
            terms=specification["terms"],
            kappa=specification["kappa"],
        )
        short_gate = specification["label"] in {"E1", "E3"}
        e4_gate = specification["label"].startswith("E4")
        config = DiagonalPathConfig(
            basis_order=specification["basis_order"],
            particles=specification["particles"],
            terms=specification["terms"],
            kappa=specification["kappa"],
            steps=(2 if short_gate else (max(1, args.steps // 2) if e4_gate else args.steps)),
            learning_rate=(5e-3 if e4_gate else 1e-2),
            final_learning_rate=1e-4,
            seed=args.seed,
            device=args.device,
            record_points=10,
            checkpoint_every=max(1, args.steps),
        )
        cached = existing.get(point_id)
        if (
            cached is not None
            and cached.get("completed")
            and cached.get("config") == asdict(config)
        ):
            result = cached
        else:
            result = run_diagonal_path_variable_projection(
                config,
                checkpoint_path=args.checkpoint_dir / f"{point_id}_checkpoint.pt",
            )
        result["point_id"] = point_id
        result["gate_label"] = specification["label"]
        points.append(result)
        print(
            f"{point_id}: energy={result['energy']:.12f} "
            f"continuum_error={result['error_vs_continuum']:.3e} "
            f"finite_error={result['error_vs_finite_basis']:.3e} "
            f"variance={result['energy_variance']:.3e}",
            flush=True,
        )

    artifact = {
        "schema_version": 1,
        "experiment": "phase28_diagonal_path_femps_ladder",
        "evidence_level": "numerical",
        "state_definition": (
            "first-quantized continuous functional-basis diagonal-path "
            "matrix-wedge FEMPS"
        ),
        "primary_route": "K nonbranching nonorthogonal Slater paths",
        "forbidden_materialization_in_production": True,
        "points": points,
    }
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "points": len(points)}, indent=2))


if __name__ == "__main__":
    main()
