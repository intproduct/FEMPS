"""Run the first functional-MPS reproduction checkpoint from arXiv:2201.12823."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from femps.baselines.training import BaselineConfig, run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--D", type=int, default=8)
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=-0.5)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-final", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device or 'auto' for the CUDA device with highest compute capability",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_baseline(
        BaselineConfig(
            num_oscillators=args.N,
            basis_order=args.D,
            bond_dimension=args.chi,
            gamma=args.gamma,
            omega=args.omega,
            steps=args.steps,
            learning_rate=args.lr,
            final_learning_rate=args.lr_final,
            seed=args.seed,
            device=args.device,
        )
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
