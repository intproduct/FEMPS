"""Phase 19 matched N=8 training comparison across Fourier MPO bonds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from femps.algorithms.ordered_continuous_training import (
    OrderedContinuousTrainingConfig,
    train_ordered_continuous_mps,
)
from femps.baselines.ordered_continuous_fourier import (
    ordered_continuous_fourier_hamiltonian_compressed_mpo,
)
from femps.devices import resolve_device


MPO_BONDS = (128, 160, 192)
REFERENCE_MPO_BOND = 192
PRODUCTION_STAGES = (
    (300, 0.01, "adam"),
    (500, 0.003, "adam"),
    (500, 0.001, "adam"),
    (300, 0.0003, "adam"),
)
REFERENCE_ENERGY_SPREAD_TOLERANCE = 2e-4
OWN_VS_REFERENCE_ENERGY_TOLERANCE = 1e-6
PEAK_CUDA_MEMORY_BUDGET_BYTES = 2 * 1024**3


def _configuration(maximum_bond: int, device: torch.device):
    return OrderedContinuousTrainingConfig(
        particles=8,
        basis_order=10,
        distance_length=0.5,
        distance_basis="multiscale_odd_hermite",
        distance_basis_scale_ratio=2.5,
        interaction_method="fourier_bessel",
        fourier_order=96,
        interaction_quadrature_order=192,
        mpo_max_bond=maximum_bond,
        bond_dimension=32,
        steps=sum(stage[0] for stage in PRODUCTION_STAGES),
        learning_rate=PRODUCTION_STAGES[0][1],
        optimization_stages=PRODUCTION_STAGES,
        seed=1862,
        projection="tensor_norm",
        device=str(device),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase19_mpo_bond_training.json"
        ),
    )
    arguments = parser.parse_args()
    device = resolve_device(arguments.device)
    if device.type != "cuda":
        raise RuntimeError("the matched N=8 training audit requires CUDA")
    gpu_name = torch.cuda.get_device_name(device)

    retained = []
    points = []
    for maximum_bond in MPO_BONDS:
        config = _configuration(maximum_bond, device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        mps, diagnostics = train_ordered_continuous_mps(config)
        torch.cuda.synchronize(device)
        point = {
            "maximum_mpo_bond": maximum_bond,
            "seed": config.seed,
            "optimization_stages": diagnostics["optimization_stages"],
            "initial_energy": diagnostics["initial_energy"],
            "own_mpo_final_energy": diagnostics["final_energy"],
            "actual_mps_maximum_bond": diagnostics["max_bond"],
            "mpo_compression_ranks": diagnostics["mpo_compression_ranks"],
            "mpo_local_discarded_norm_not_global_certificate": diagnostics[
                "mpo_compression_local_discarded_norm"
            ],
            "dense_raw_fourier_bulk_materialized": diagnostics[
                "dense_raw_fourier_bulk_materialized"
            ],
            "maximum_mpo_build_intermediate_tensor_elements": diagnostics[
                "maximum_mpo_build_intermediate_tensor_elements"
            ],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
        }
        points.append(point)
        retained.append(mps)

    reference_mpo, reference_build = (
        ordered_continuous_fourier_hamiltonian_compressed_mpo(
            8,
            10,
            0.5,
            96,
            REFERENCE_MPO_BOND,
            distance_basis="multiscale_odd_hermite",
            distance_scale_ratio=2.5,
            local_quadrature_order=192,
            device=device,
        )
    )
    for point, mps in zip(points, retained, strict=True):
        reference_energy = float(mps.energy_with_MPO(reference_mpo).detach())
        point["bond_192_reference_energy_on_same_trained_state"] = (
            reference_energy
        )
        point["own_vs_reference_energy_absolute_difference"] = abs(
            point["own_mpo_final_energy"] - reference_energy
        )
        point["own_vs_reference_energy_pass"] = (
            point["own_vs_reference_energy_absolute_difference"]
            < OWN_VS_REFERENCE_ENERGY_TOLERANCE
        )
        point["memory_budget_pass"] = (
            point["peak_cuda_memory_bytes"]
            < PEAK_CUDA_MEMORY_BUDGET_BYTES
        )
    reference_energies = [
        point["bond_192_reference_energy_on_same_trained_state"]
        for point in points
    ]
    reference_energy_spread = max(reference_energies) - min(reference_energies)
    record = {
        "schema_version": 1,
        "experiment": "phase19_matched_n8_mpo_bond_training",
        "device": str(device),
        "gpu": gpu_name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": "float64",
        "predeclared_configuration": {
            "mpo_bonds": MPO_BONDS,
            "reference_mpo_bond": REFERENCE_MPO_BOND,
            "seed": 1862,
            "optimization_stages": PRODUCTION_STAGES,
            "reference_energy_spread_tolerance": (
                REFERENCE_ENERGY_SPREAD_TOLERANCE
            ),
            "own_vs_reference_energy_tolerance": (
                OWN_VS_REFERENCE_ENERGY_TOLERANCE
            ),
            "peak_cuda_memory_budget_bytes": (
                PEAK_CUDA_MEMORY_BUDGET_BYTES
            ),
        },
        "reference_mpo": {
            "maximum_bond": REFERENCE_MPO_BOND,
            "retained_ranks": list(reference_build["retained_ranks"]),
            "dense_raw_fourier_bulk_materialized": reference_build[
                "dense_raw_fourier_bulk_materialized"
            ],
        },
        "points": points,
        "bond_192_reference_energy_spread_across_matched_runs": (
            reference_energy_spread
        ),
        "reference_energy_spread_pass": (
            reference_energy_spread < REFERENCE_ENERGY_SPREAD_TOLERANCE
        ),
    }
    record["matched_training_pass"] = (
        record["reference_energy_spread_pass"]
        and all(point["own_vs_reference_energy_pass"] for point in points)
        and all(point["memory_budget_pass"] for point in points)
        and not any(
            point["dense_raw_fourier_bulk_materialized"] for point in points
        )
    )
    record["selected_smallest_passing_production_mpo_bond"] = (
        min(MPO_BONDS) if record["matched_training_pass"] else None
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0 if record["matched_training_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
