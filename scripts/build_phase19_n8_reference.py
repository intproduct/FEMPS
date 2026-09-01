"""Combine independently computed D=14 exterior points into Phase 19 record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _single_point(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    points = record["basis_scan"]
    if len(points) != 1:
        raise ValueError(f"{path} must contain exactly one basis point")
    point = points[0]
    if point["N"] != 8 or point["D"] != 14:
        raise ValueError(f"{path} is not an N=8,D=14 exterior point")
    return point


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q128", type=Path, required=True)
    parser.add_argument("--q160", type=Path, required=True)
    parser.add_argument(
        "--prior-d12",
        type=Path,
        default=Path(
            "docs/experiments/results/soft_coulomb_n8_truth_sweep.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase19_n8_exterior_d14.json"
        ),
    )
    arguments = parser.parse_args()
    q128 = _single_point(arguments.q128)
    q160 = _single_point(arguments.q160)
    if q128["Q"] != 128 or q160["Q"] != 160:
        raise ValueError("source points must have Q=128 and Q=160")
    prior = json.loads(arguments.prior_d12.read_text(encoding="utf-8"))
    d12_q160 = next(
        point
        for point in prior["quadrature_scan_at_largest_D"]
        if point["D"] == 12 and point["Q"] == 160
    )
    record = {
        "schema_version": 1,
        "experiment": "phase19_n8_exterior_D14_reference_extension",
        "hamiltonian": "spin-polarized N=8, g=a=omega=1 on R",
        "truth_path": "direct four-index Slater-Condon exterior Hamiltonian",
        "reference_is_numerical_not_continuum_bound": True,
        "source_records": {
            "D12": str(arguments.prior_d12),
            "D14_Q128": str(arguments.q128),
            "D14_Q160": str(arguments.q160),
        },
        "D12_Q160": d12_q160,
        "D14_quadrature_scan": [q128, q160],
        "D14_Q128_minus_Q160": (
            q128["ground_energy"] - q160["ground_energy"]
        ),
        "D12_Q160_minus_D14_Q160": (
            d12_q160["ground_energy"] - q160["ground_energy"]
        ),
        "strongest_numerical_reference": {
            "basis_order": 14,
            "quadrature_order": 160,
            "ground_energy": q160["ground_energy"],
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
