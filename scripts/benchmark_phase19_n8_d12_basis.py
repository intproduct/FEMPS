"""Phase 19 blind N=8,D=12 multiscale basis refinement audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from femps.algorithms.ordered_continuous_training import (
    OrderedContinuousTrainingConfig,
    train_ordered_continuous_mps,
)
from femps.baselines.ordered_continuous_fourier import (
    ordered_continuous_fourier_hamiltonian_compressed_mpo,
)
from femps.basis.multiscale_odd_hermite import (
    multiscale_odd_hermite_basis_values,
    multiscale_odd_hermite_condition_number,
)
from femps.devices import resolve_device


N8_EXTERIOR_D14_REFERENCE = 44.445670415298615
GATE_F_D10_BEST_ENERGY = 44.45437326357539
SCALE_CANDIDATES = tuple(
    (scale, ratio)
    for scale in [0.40, 0.45, 0.50, 0.55, 0.60]
    for ratio in [2.5, 3.0]
)
PRODUCTION_STAGES = (
    (300, 0.01, "adam"),
    (500, 0.003, "adam"),
    (500, 0.001, "adam"),
    (300, 0.0003, "adam"),
)
REFERENCE_AGREEMENT_TOLERANCE = 1.2e-2
LOCAL_OPTIMIZER_CONSISTENCY_TOLERANCE = 1e-3
FOURIER_96_VS_112_TOLERANCE = 5e-6
QUADRATURE_192_VS_224_TOLERANCE = 1e-8
PEAK_CUDA_MEMORY_BUDGET_BYTES = 2 * 1024**3


def _configuration(
    scale: float,
    ratio: float,
    *,
    bond: int,
    seed: int,
    stages,
    device: torch.device,
) -> OrderedContinuousTrainingConfig:
    return OrderedContinuousTrainingConfig(
        particles=8,
        basis_order=12,
        distance_length=scale,
        distance_basis="multiscale_odd_hermite",
        distance_basis_scale_ratio=ratio,
        interaction_method="fourier_bessel",
        fourier_order=96,
        interaction_quadrature_order=224,
        mpo_max_bond=128,
        bond_dimension=bond,
        steps=sum(stage[0] for stage in stages),
        learning_rate=stages[0][1],
        optimization_stages=stages,
        seed=seed,
        projection="tensor_norm",
        device=str(device),
    )


def _run(config: OrderedContinuousTrainingConfig, *, retain_state: bool):
    device = torch.device(config.device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    mps, diagnostics = train_ordered_continuous_mps(config)
    torch.cuda.synchronize(device)
    record = {
        "scale": config.distance_length,
        "scale_ratio": config.distance_basis_scale_ratio,
        "seed": config.seed,
        "requested_mps_bond": config.bond_dimension,
        "optimization_stages": diagnostics["optimization_stages"],
        "initial_energy": diagnostics["initial_energy"],
        "final_energy": diagnostics["final_energy"],
        "actual_mps_maximum_bond": diagnostics["max_bond"],
        "mpo_compression_ranks": diagnostics["mpo_compression_ranks"],
        "dense_raw_fourier_bulk_materialized": diagnostics[
            "dense_raw_fourier_bulk_materialized"
        ],
        "maximum_mpo_build_intermediate_tensor_elements": diagnostics[
            "maximum_mpo_build_intermediate_tensor_elements"
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    return record, mps if retain_state else None


def _basis_overlap_control(scale: float, ratio: float) -> dict[str, object]:
    order = 12
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(1000)
    maximum_mode = (order + 1) // 2 - 1
    cutoff = scale * math.sqrt(ratio) * (
        math.sqrt(4 * (maximum_mode + 1) + 2) + 10
    )
    nodes = torch.from_numpy((raw_nodes + 1) * cutoff / 2)
    weights = torch.from_numpy(raw_weights * cutoff / 2)
    basis = multiscale_odd_hermite_basis_values(
        order, nodes, scale, ratio
    )
    overlap = torch.einsum("xm,x,xn->mn", basis, weights, basis)
    boundary = multiscale_odd_hermite_basis_values(
        order, torch.zeros(1, dtype=torch.float64), scale, ratio
    )
    return {
        "basis_order": order,
        "scale": scale,
        "scale_ratio": ratio,
        "primitive_overlap_condition_number": (
            multiscale_odd_hermite_condition_number(order, scale, ratio)
        ),
        "independent_quadrature_points": 1000,
        "orthonormality_maximum_absolute_residual": float(
            torch.max(torch.abs(overlap - torch.eye(order)))
        ),
        "collision_boundary_maximum_absolute_value": float(
            torch.max(torch.abs(boundary))
        ),
    }


def _operator_energy(
    mps,
    scale: float,
    ratio: float,
    *,
    fourier_order: int,
    quadrature_order: int,
    device: torch.device,
) -> dict[str, object]:
    mpo, build = ordered_continuous_fourier_hamiltonian_compressed_mpo(
        8,
        12,
        scale,
        fourier_order,
        160,
        distance_basis="multiscale_odd_hermite",
        distance_scale_ratio=ratio,
        local_quadrature_order=quadrature_order,
        device=device,
    )
    return {
        "fourier_order": fourier_order,
        "local_quadrature_order": quadrature_order,
        "mpo_maximum_bond": 160,
        "energy_on_fixed_production_state": float(
            mps.energy_with_MPO(mpo).detach()
        ),
        "dense_raw_fourier_bulk_materialized": build[
            "dense_raw_fourier_bulk_materialized"
        ],
    }


def _local_optimizer(
    mps,
    scale: float,
    ratio: float,
    device: torch.device,
) -> dict[str, object]:
    from latticetn.dmrg import run_dmrg

    mpo, _ = ordered_continuous_fourier_hamiltonian_compressed_mpo(
        8,
        12,
        scale,
        96,
        128,
        distance_basis="multiscale_odd_hermite",
        distance_scale_ratio=ratio,
        local_quadrature_order=224,
        device=device,
    )
    initial_energy = float(mps.energy_with_MPO(mpo).detach())
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = run_dmrg(
        mps,
        mpo,
        chi=32,
        num_sweeps=2,
        seed=1942,
        solver="lanczos",
        lanczos_kwargs={
            "max_iter": 30,
            "tol": 1e-8,
            "num_restarts": 1,
            "seed": 1942,
        },
    )
    torch.cuda.synchronize(device)
    return {
        "initial_energy": initial_energy,
        "history": result["history"],
        "final_energy": result["final_energy"],
        "absolute_difference_vs_global_AD": abs(
            result["final_energy"] - initial_energy
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase19_n8_d12_multiscale.json"
        ),
    )
    arguments = parser.parse_args()
    device = resolve_device(arguments.device)
    if device.type != "cuda":
        raise RuntimeError("the N=8,D=12 basis audit requires CUDA")
    gpu_name = torch.cuda.get_device_name(device)

    scale_scan = []
    for scale, ratio in SCALE_CANDIDATES:
        point, _ = _run(
            _configuration(
                scale,
                ratio,
                bond=16,
                seed=1940,
                stages=((250, 0.01, "adam"),),
                device=device,
            ),
            retain_state=False,
        )
        scale_scan.append(point)
    selected = min(scale_scan, key=lambda point: point["final_energy"])
    selected_scale = selected["scale"]
    selected_ratio = selected["scale_ratio"]
    production, mps = _run(
        _configuration(
            selected_scale,
            selected_ratio,
            bond=32,
            seed=1941,
            stages=PRODUCTION_STAGES,
            device=device,
        ),
        retain_state=True,
    )

    # Operator/reference audits begin only after the blind basis choice and
    # production optimization are complete.
    basis_control = _basis_overlap_control(selected_scale, selected_ratio)
    fourier_scan = [
        _operator_energy(
            mps,
            selected_scale,
            selected_ratio,
            fourier_order=order,
            quadrature_order=224,
            device=device,
        )
        for order in [80, 96, 112]
    ]
    quadrature_scan = [
        _operator_energy(
            mps,
            selected_scale,
            selected_ratio,
            fourier_order=96,
            quadrature_order=order,
            device=device,
        )
        for order in [160, 192, 224]
    ]
    local_optimizer = _local_optimizer(
        mps, selected_scale, selected_ratio, device
    )
    reference_error = (
        production["final_energy"] - N8_EXTERIOR_D14_REFERENCE
    )
    gate_f_d10_error = (
        GATE_F_D10_BEST_ENERGY - N8_EXTERIOR_D14_REFERENCE
    )
    fourier_96 = next(
        point for point in fourier_scan if point["fourier_order"] == 96
    )
    fourier_112 = next(
        point for point in fourier_scan if point["fourier_order"] == 112
    )
    quadrature_192 = next(
        point
        for point in quadrature_scan
        if point["local_quadrature_order"] == 192
    )
    quadrature_224 = next(
        point
        for point in quadrature_scan
        if point["local_quadrature_order"] == 224
    )
    record = {
        "schema_version": 1,
        "experiment": "phase19_blind_n8_D12_multiscale_refinement",
        "device": str(device),
        "gpu": gpu_name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": "float64",
        "predeclared_configuration": {
            "scale_candidates": SCALE_CANDIDATES,
            "blind_scale_scan_steps": 250,
            "blind_scale_scan_mps_bond": 16,
            "production_stages": PRODUCTION_STAGES,
            "production_mps_bond": 32,
            "production_mpo_bond": 128,
            "reference_agreement_tolerance": (
                REFERENCE_AGREEMENT_TOLERANCE
            ),
            "local_optimizer_consistency_tolerance": (
                LOCAL_OPTIMIZER_CONSISTENCY_TOLERANCE
            ),
            "fourier_96_vs_112_tolerance": (
                FOURIER_96_VS_112_TOLERANCE
            ),
            "quadrature_192_vs_224_tolerance": (
                QUADRATURE_192_VS_224_TOLERANCE
            ),
            "peak_cuda_memory_budget_bytes": (
                PEAK_CUDA_MEMORY_BUDGET_BYTES
            ),
        },
        "blind_scale_scan": scale_scan,
        "selected_scale": selected_scale,
        "selected_scale_ratio": selected_ratio,
        "production": production,
        "basis_control": basis_control,
        "fixed_state_fourier_scan": fourier_scan,
        "fixed_state_local_quadrature_scan": quadrature_scan,
        "independent_chi32_local_optimizer": local_optimizer,
        "exterior_D14_Q160_numerical_reference": (
            N8_EXTERIOR_D14_REFERENCE
        ),
        "reference_is_numerical_not_continuum_bound": True,
        "production_error_vs_exterior_D14_reference": reference_error,
        "gate_f_D10_error_vs_same_D14_reference": gate_f_d10_error,
        "basis_error_reduction_fraction_vs_gate_f_D10": (
            1 - reference_error / gate_f_d10_error
        ),
        "fourier_96_vs_112_energy_absolute_difference": abs(
            fourier_96["energy_on_fixed_production_state"]
            - fourier_112["energy_on_fixed_production_state"]
        ),
        "quadrature_192_vs_224_energy_absolute_difference": abs(
            quadrature_192["energy_on_fixed_production_state"]
            - quadrature_224["energy_on_fixed_production_state"]
        ),
    }
    record["reference_agreement_pass"] = (
        abs(reference_error) < REFERENCE_AGREEMENT_TOLERANCE
    )
    record["basis_improves_vs_gate_f_D10"] = (
        abs(reference_error) < abs(gate_f_d10_error)
    )
    record["local_optimizer_consistency_pass"] = (
        local_optimizer["absolute_difference_vs_global_AD"]
        < LOCAL_OPTIMIZER_CONSISTENCY_TOLERANCE
    )
    record["fourier_convergence_pass"] = (
        record["fourier_96_vs_112_energy_absolute_difference"]
        < FOURIER_96_VS_112_TOLERANCE
    )
    record["local_quadrature_convergence_pass"] = (
        record["quadrature_192_vs_224_energy_absolute_difference"]
        < QUADRATURE_192_VS_224_TOLERANCE
    )
    record["memory_budget_pass"] = (
        production["peak_cuda_memory_bytes"] < PEAK_CUDA_MEMORY_BUDGET_BYTES
        and local_optimizer["peak_cuda_memory_bytes"]
        < PEAK_CUDA_MEMORY_BUDGET_BYTES
    )
    record["n8_D12_refinement_pass"] = all(
        record[key]
        for key in [
            "reference_agreement_pass",
            "basis_improves_vs_gate_f_D10",
            "local_optimizer_consistency_pass",
            "fourier_convergence_pass",
            "local_quadrature_convergence_pass",
            "memory_budget_pass",
        ]
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0 if record["n8_D12_refinement_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
