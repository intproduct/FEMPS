"""Run the first functional Pfaffian fermion benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from femps.algorithms import PfaffianPairConfig, run_pfaffian_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=12)
    parser.add_argument("--kappa", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/results/fermion_e1_e2"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, kappa in (("E1", 0.0), ("E2", args.kappa)):
        config = PfaffianPairConfig(
            basis_order=args.basis_order,
            kappa=kappa,
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        )
        checkpoint = args.output_dir / f"{name.lower()}_checkpoint.pt"
        result = run_pfaffian_pair(config, checkpoint_path=checkpoint)
        (args.output_dir / f"{name.lower()}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        results[name] = result
        print(
            f"{name}: final={result['final_energy']:.12f} "
            f"truncated_error={result['error_vs_truncated']:.3e}",
            flush=True,
        )
    summary = {
        "schema_version": 1,
        "experiment": "functional_pfaffian_e1_e2",
        "E1": results["E1"],
        "E2": results["E2"],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
