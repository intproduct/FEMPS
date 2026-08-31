"""Run reproducible D, chi, and seed scans for the 2201 baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from femps.baselines.training import BaselineConfig, run_baseline


def _integers(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument(
        "--D-values", type=_integers, default=_integers("2,4,6,8,10,12")
    )
    parser.add_argument(
        "--chi-values", type=_integers, default=_integers("2,4,8,12,16,20")
    )
    parser.add_argument("--seeds", type=_integers, default=_integers("1,2,3"))
    parser.add_argument("--anchor-D", type=int, default=8)
    parser.add_argument("--anchor-chi", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=-0.5)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-final", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/results/2201_sweep"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _point_path(output_dir: Path, axis: str, config: BaselineConfig) -> Path:
    return output_dir / "points" / (
        f"N{config.num_oscillators}_D{config.basis_order}_"
        f"chi{config.bond_dimension}_seed{config.seed}.json"
    )


def _run_point(
    output_dir: Path,
    axis: str,
    config: BaselineConfig,
    *,
    resume: bool,
) -> dict:
    path = _point_path(output_dir, axis, config)
    if resume and path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("config") == asdict(config):
            print(f"resume {path}", flush=True)
            return record
    print(
        f"run {axis}: D={config.basis_order} "
        f"chi={config.bond_dimension} seed={config.seed}",
        flush=True,
    )
    record = run_baseline(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _summary_row(record: dict) -> dict:
    config = record["config"]
    return {
        "D": config["basis_order"],
        "chi": config["bond_dimension"],
        "seed": config["seed"],
        "final_energy": record["final_energy"],
        "absolute_error": record["absolute_error"],
        "variational_margin": record["variational_margin"],
        "elapsed_seconds": record["elapsed_seconds"],
        "parameter_count": record["parameter_count"],
        "peak_cuda_memory_bytes": record["peak_cuda_memory_bytes"],
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_cache: dict[BaselineConfig, dict] = {}

    def config(*, basis_order: int, bond_dimension: int, seed: int) -> BaselineConfig:
        return BaselineConfig(
            num_oscillators=args.N,
            basis_order=basis_order,
            bond_dimension=bond_dimension,
            gamma=args.gamma,
            omega=args.omega,
            steps=args.steps,
            learning_rate=args.lr,
            final_learning_rate=args.lr_final,
            seed=seed,
            device=args.device,
        )

    def point(axis: str, point_config: BaselineConfig) -> dict:
        if point_config not in record_cache:
            record_cache[point_config] = _run_point(
                args.output_dir,
                axis,
                point_config,
                resume=args.resume,
            )
        return record_cache[point_config]

    basis_records = [
        point(
            "basis",
            config(basis_order=D, bond_dimension=args.anchor_chi, seed=0),
        )
        for D in args.D_values
    ]
    bond_records = [
        point(
            "bond",
            config(basis_order=args.anchor_D, bond_dimension=chi, seed=0),
        )
        for chi in args.chi_values
    ]
    seed_records = [
        point(
            "seed",
            config(
                basis_order=args.anchor_D,
                bond_dimension=args.anchor_chi,
                seed=seed,
            ),
        )
        for seed in args.seeds
    ]
    summary = {
        "schema_version": 1,
        "experiment": "2201_two_body_functional_mps_sweep",
        "scan": {
            "N": args.N,
            "D_values": args.D_values,
            "chi_values": args.chi_values,
            "seeds": args.seeds,
            "anchor_D": args.anchor_D,
            "anchor_chi": args.anchor_chi,
            "gamma": args.gamma,
            "omega": args.omega,
            "steps": args.steps,
            "learning_rate": args.lr,
            "final_learning_rate": args.lr_final,
            "device": args.device,
        },
        "basis_sweep": [_summary_row(record) for record in basis_records],
        "bond_sweep": [_summary_row(record) for record in bond_records],
        "seed_sweep": [_summary_row(record) for record in seed_records],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
