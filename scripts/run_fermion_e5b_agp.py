"""Blind finite-AGP seed scan for six interacting harmonic fermions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from femps.algorithms import FiniteAgpConfig, run_finite_agp_variable_projection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=10)
    parser.add_argument("--terms", type=int, default=1)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/fermion_e5b_single_agp.json"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("docs/experiments/results/e5b_checkpoints"),
    )
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "experiment": "fermion_e5b_six_particle_blind_agp_sweep",
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
            particles=6,
            agp_terms=args.terms,
            kappa=args.kappa,
            steps=args.steps,
            learning_rate=1e-2,
            final_learning_rate=1e-5,
            seed=seed,
            device=args.device,
            record_points=20,
            checkpoint_every=100,
        )
        checkpoint = args.checkpoint_directory / (
            f"e5b_k{args.terms}_seed{seed}_checkpoint.pt"
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"seed={seed} E={run['final_energy']:.12f} "
            f"finite_error={run['error_vs_finite_basis']:.3e}"
        )


if __name__ == "__main__":
    main()
