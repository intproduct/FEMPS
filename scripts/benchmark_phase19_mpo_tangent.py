"""Phase 19 gauge-fixed physical tangent audit for Fourier MPO bonds."""

from __future__ import annotations

import argparse
import json
import math
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
from femps.benchmarks.mps_tangent import (
    left_gauge_physical_tangent_directions,
    mpo_energy_and_tangent_directional_derivatives,
)
from femps.devices import resolve_device


PRODUCTION_STAGES = (
    (300, 0.01, "adam"),
    (500, 0.003, "adam"),
    (500, 0.001, "adam"),
    (300, 0.0003, "adam"),
)
MPO_BONDS = (128, 160, 192)
REFERENCE_MPO_BOND = 192
ENERGY_TOLERANCE = 1e-6
TANGENT_MAXIMUM_ABSOLUTE_TOLERANCE = 1e-6
TANGENT_RELATIVE_L2_TOLERANCE = 5e-5
TANGENT_COSINE_MINIMUM = 1 - 1e-8


def _production_state(device: torch.device):
    config = OrderedContinuousTrainingConfig(
        particles=8,
        basis_order=10,
        distance_length=0.5,
        distance_basis="multiscale_odd_hermite",
        distance_basis_scale_ratio=2.5,
        interaction_method="fourier_bessel",
        fourier_order=96,
        interaction_quadrature_order=192,
        mpo_max_bond=128,
        bond_dimension=32,
        steps=sum(stage[0] for stage in PRODUCTION_STAGES),
        learning_rate=PRODUCTION_STAGES[0][1],
        optimization_stages=PRODUCTION_STAGES,
        seed=1862,
        projection="tensor_norm",
        device=str(device),
    )
    started = time.perf_counter()
    mps, diagnostics = train_ordered_continuous_mps(config)
    torch.cuda.synchronize(device)
    return mps, {
        "source": "reproduced_phase18_best_blind_seed",
        "seed": config.seed,
        "optimization_stages": diagnostics["optimization_stages"],
        "initial_energy": diagnostics["initial_energy"],
        "final_energy": diagnostics["final_energy"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def _checkpoint_state(checkpoint_path: Path, device: torch.device):
    from latticetn.mps import MPS

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    return (
        MPS.from_tensors(
            checkpoint["tensors"],
            dtype=torch.float64,
            device=device,
            requires_grad=False,
        ),
        {
            "source": "development_only_phase18_ignored_checkpoint",
            "checkpoint": str(checkpoint_path),
        },
    )


def _comparison(observed: dict, reference: dict) -> dict[str, object]:
    observed_derivatives = torch.tensor(
        observed["directional_derivatives"], dtype=torch.float64
    )
    reference_derivatives = torch.tensor(
        reference["directional_derivatives"], dtype=torch.float64
    )
    difference = observed_derivatives - reference_derivatives
    observed_norm = torch.linalg.vector_norm(observed_derivatives)
    reference_norm = torch.linalg.vector_norm(reference_derivatives)
    energy_difference = abs(observed["energy"] - reference["energy"])
    maximum_absolute = float(torch.max(torch.abs(difference)))
    relative_l2 = float(torch.linalg.vector_norm(difference) / reference_norm)
    cosine = float(
        torch.dot(observed_derivatives, reference_derivatives)
        / (observed_norm * reference_norm)
    )
    result = {
        "observed_mpo_bond": observed["maximum_bond"],
        "reference_mpo_bond": reference["maximum_bond"],
        "energy_absolute_difference": energy_difference,
        "tangent_derivative_maximum_absolute_difference": maximum_absolute,
        "tangent_derivative_relative_l2_difference": relative_l2,
        "tangent_derivative_cosine_similarity": cosine,
    }
    result["energy_pass"] = energy_difference < ENERGY_TOLERANCE
    result["tangent_maximum_absolute_pass"] = (
        maximum_absolute < TANGENT_MAXIMUM_ABSOLUTE_TOLERANCE
    )
    result["tangent_relative_l2_pass"] = (
        relative_l2 < TANGENT_RELATIVE_L2_TOLERANCE
    )
    result["tangent_cosine_pass"] = cosine > TANGENT_COSINE_MINIMUM
    result["all_pass"] = all(
        result[key]
        for key in [
            "energy_pass",
            "tangent_maximum_absolute_pass",
            "tangent_relative_l2_pass",
            "tangent_cosine_pass",
        ]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="development shortcut; formal records omit this option",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase19_mpo_tangent_audit.json"
        ),
    )
    arguments = parser.parse_args()
    device = resolve_device(arguments.device)
    if device.type != "cuda":
        raise RuntimeError("the N=8 tangent audit requires CUDA")
    gpu_name = torch.cuda.get_device_name(device)
    if arguments.checkpoint is None:
        mps, source_record = _production_state(device)
        formal_record = True
    else:
        mps, source_record = _checkpoint_state(arguments.checkpoint, device)
        formal_record = False

    canonical, directions = left_gauge_physical_tangent_directions(
        mps,
        directions_per_site=2,
        seed=1914,
    )
    direction_metadata = [
        {key: value for key, value in direction.items() if key != "tensor"}
        for direction in directions
    ]
    points = []
    for maximum_bond in MPO_BONDS:
        mpo, build = ordered_continuous_fourier_hamiltonian_compressed_mpo(
            8,
            10,
            0.5,
            96,
            maximum_bond,
            distance_basis="multiscale_odd_hermite",
            distance_scale_ratio=2.5,
            local_quadrature_order=192,
            device=device,
        )
        energy, derivatives, raw_gradient_norm = (
            mpo_energy_and_tangent_directional_derivatives(
                canonical, mpo, directions
            )
        )
        points.append(
            {
                "maximum_bond": maximum_bond,
                "energy": energy,
                "directional_derivatives": derivatives,
                "directional_derivative_l2_norm": math.sqrt(
                    sum(value**2 for value in derivatives)
                ),
                "raw_parameter_gradient_norm_diagnostic": raw_gradient_norm,
                "retained_ranks": list(build["retained_ranks"]),
                "local_discarded_norm_not_global_certificate": float(
                    build[
                        "local_discarded_norm_not_global_certificate"
                    ]
                ),
                "dense_raw_fourier_bulk_materialized": build[
                    "dense_raw_fourier_bulk_materialized"
                ],
            }
        )
    reference = next(
        point
        for point in points
        if point["maximum_bond"] == REFERENCE_MPO_BOND
    )
    comparisons = [
        _comparison(point, reference)
        for point in points
        if point["maximum_bond"] != REFERENCE_MPO_BOND
    ]
    passing_bonds = [
        point["observed_mpo_bond"]
        for point in comparisons
        if point["all_pass"]
    ]
    record = {
        "schema_version": 1,
        "experiment": "phase19_gauge_fixed_physical_tangent_mpo_audit",
        "formal_record": formal_record,
        "device": str(device),
        "gpu": gpu_name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": "float64",
        "predeclared_configuration": {
            "mpo_bonds": MPO_BONDS,
            "reference_mpo_bond": REFERENCE_MPO_BOND,
            "directions_per_nonzero_site_block": 2,
            "direction_seed": 1914,
            "energy_tolerance": ENERGY_TOLERANCE,
            "tangent_maximum_absolute_tolerance": (
                TANGENT_MAXIMUM_ABSOLUTE_TOLERANCE
            ),
            "tangent_relative_l2_tolerance": (
                TANGENT_RELATIVE_L2_TOLERANCE
            ),
            "tangent_cosine_minimum": TANGENT_COSINE_MINIMUM,
        },
        "source_state": source_record,
        "tangent_construction": {
            "gauge": "left_canonical_A_transpose_B_equals_zero",
            "normalization": "native_many_body_MPS_tangent_norm_equals_one",
            "directions": direction_metadata,
            "maximum_state_overlap_absolute_value": max(
                direction["state_overlap_absolute_value"]
                for direction in direction_metadata
            ),
            "maximum_normalized_physical_norm_error": max(
                abs(direction["normalized_physical_norm"] - 1)
                for direction in direction_metadata
            ),
        },
        "points": points,
        "comparisons_vs_bond_192": comparisons,
        "smallest_passing_mpo_bond": (
            min(passing_bonds) if passing_bonds else None
        ),
    }
    record["tangent_audit_pass"] = bool(passing_bonds)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0 if record["tangent_audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
