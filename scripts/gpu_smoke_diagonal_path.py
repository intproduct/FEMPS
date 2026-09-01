"""Reproduce the Phase 28 Blackwell resource/parity point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from femps.algorithms import (
    DiagonalPathConfig,
    run_diagonal_path_variable_projection,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--cpu-artifact",
        type=Path,
        default=Path(
            "docs/experiments/results/phase28_diagonal_path_ladder.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase28_diagonal_path_gpu_parity.json"
        ),
    )
    args = parser.parse_args()
    config = DiagonalPathConfig(
        basis_order=6,
        particles=4,
        terms=4,
        kappa=0.35,
        steps=100,
        learning_rate=5e-3,
        final_learning_rate=1e-4,
        seed=17,
        device=args.device,
        record_points=10,
        checkpoint_every=100,
    )
    gpu_result = run_diagonal_path_variable_projection(config)
    cpu_artifact = json.loads(args.cpu_artifact.read_text(encoding="utf-8"))
    cpu_point = next(
        point
        for point in cpu_artifact["points"]
        if point["point_id"] == "N4_D6_K4_kappa0.35"
    )
    artifact = {
        "schema_version": 1,
        "experiment": "phase28_diagonal_path_blackwell_parity",
        "evidence_level": "numerical",
        "cpu_point_id": cpu_point["point_id"],
        "cpu_energy": cpu_point["energy"],
        "gpu_energy": gpu_result["energy"],
        "absolute_energy_difference": abs(
            cpu_point["energy"] - gpu_result["energy"]
        ),
        "gpu_result": gpu_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
