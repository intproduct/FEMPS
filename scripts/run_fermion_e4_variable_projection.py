"""Run resumable blind finite-AGP variable-projection E4 seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from femps.algorithms import FiniteAgpConfig, run_finite_agp_variable_projection


def _write(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=8)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--terms", type=int, default=2)
    parser.add_argument("--kappa", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/fermion_e4_variable_projection.json"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("docs/experiments/results"),
    )
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "experiment": "fermion_e4_blind_variable_projection_seed_sweep",
        "runs": [],
    }
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    completed = {
        run["config"]["seed"]
        for run in result["runs"]
        if run.get("completed", False)
    }
    for seed in args.seeds:
        if seed in completed:
            print(f"skip completed seed={seed}")
            continue
        config = FiniteAgpConfig(
            basis_order=args.basis_order,
            particles=args.particles,
            agp_terms=args.terms,
            kappa=args.kappa,
            steps=args.steps,
            learning_rate=args.learning_rate,
            final_learning_rate=args.final_learning_rate,
            seed=seed,
            device=args.device,
            record_points=20,
            checkpoint_every=100,
        )
        checkpoint = args.checkpoint_directory / (
            f"fermion_e4_variable_projection_seed{seed}_checkpoint.pt"
        )
        run = run_finite_agp_variable_projection(
            config,
            checkpoint_path=checkpoint,
            resume=args.resume and checkpoint.exists(),
        )
        result["runs"] = [
            previous
            for previous in result["runs"]
            if previous["config"]["seed"] != seed
        ]
        result["runs"].append(run)
        _write(args.output, result)
        print(
            f"seed={seed} E={run['final_energy']:.12f} "
            f"finite_error={run['error_vs_finite_basis']:.3e} "
            f"condition={run['final_retained_condition_number']:.3e}"
        )


if __name__ == "__main__":
    main()
