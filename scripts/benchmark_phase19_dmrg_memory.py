"""Phase 19 formal chi-32 DMRG bounded-intermediate resource audit."""

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


PRODUCTION_STAGES = (
    (300, 0.01, "adam"),
    (500, 0.003, "adam"),
    (500, 0.001, "adam"),
    (300, 0.0003, "adam"),
)
PEAK_CUDA_MEMORY_BUDGET_BYTES = 2 * 1024**3
SWEEP_CONSISTENCY_TOLERANCE = 1e-6


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
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    mps, diagnostics = train_ordered_continuous_mps(config)
    torch.cuda.synchronize(device)
    return mps, {
        "source": "reproduced_phase18_best_blind_seed",
        "seed": config.seed,
        "optimization_stages": diagnostics["optimization_stages"],
        "initial_energy": diagnostics["initial_energy"],
        "final_energy": diagnostics["final_energy"],
        "actual_mps_maximum_bond": diagnostics["max_bond"],
        "mpo_compression_strategy": diagnostics[
            "mpo_compression_strategy"
        ],
        "dense_raw_fourier_bulk_materialized": diagnostics[
            "dense_raw_fourier_bulk_materialized"
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _checkpoint_state(checkpoint_path: Path, device: torch.device):
    from latticetn.mps import MPS

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    mps = MPS.from_tensors(
        checkpoint["tensors"],
        dtype=torch.float64,
        device=device,
        requires_grad=False,
    )
    return mps, {
        "source": "development_only_phase18_ignored_checkpoint",
        "checkpoint": str(checkpoint_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="development shortcut; formal records omit this option",
    )
    parser.add_argument("--maximum-iterations", type=int, default=30)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase19_dmrg_contraction.json"
        ),
    )
    arguments = parser.parse_args()
    if arguments.maximum_iterations < 1 or arguments.sweeps < 1:
        raise ValueError("maximum iterations and sweeps must be positive")
    device = resolve_device(arguments.device)
    if device.type != "cuda":
        raise RuntimeError("the chi-32 memory audit requires CUDA")
    gpu_name = torch.cuda.get_device_name(device)

    from latticetn.dmrg import run_dmrg

    if arguments.checkpoint is None:
        mps, source_record = _production_state(device)
        formal_record = True
        scale = 0.5
        scale_ratio = 2.5
    else:
        mps, source_record = _checkpoint_state(arguments.checkpoint, device)
        formal_record = False
        scale = 0.5
        scale_ratio = 2.5

    mpo, build = ordered_continuous_fourier_hamiltonian_compressed_mpo(
        8,
        10,
        scale,
        96,
        128,
        distance_basis="multiscale_odd_hermite",
        distance_scale_ratio=scale_ratio,
        local_quadrature_order=192,
        device=device,
    )
    maximum_mps_bond = max(
        max(tensor.shape[0], tensor.shape[2]) for tensor in mps.tensors
    )
    maximum_mpo_bond = max(
        max(tensor.shape[0], tensor.shape[1]) for tensor in mpo.tensors
    )
    staged_temporary_element_bound = (
        maximum_mps_bond**2 * maximum_mpo_bond * mpo.dim**2
    )
    initial_energy = float(mps.energy_with_MPO(mpo).detach())
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = run_dmrg(
        mps,
        mpo,
        chi=32,
        num_sweeps=arguments.sweeps,
        seed=1901,
        solver="lanczos",
        lanczos_kwargs={
            "max_iter": arguments.maximum_iterations,
            "tol": 1e-8,
            "num_restarts": 1,
            "seed": 1901,
        },
    )
    torch.cuda.synchronize(device)
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    sweep_consistency = (
        abs(result["history"][-1]["energy"] - result["history"][-2]["energy"])
        if len(result["history"]) >= 2
        else None
    )
    record = {
        "schema_version": 1,
        "experiment": "phase19_chi32_staged_heff_resource_audit",
        "formal_record": formal_record,
        "device": str(device),
        "gpu": gpu_name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": "float64",
        "predeclared_budgets": {
            "peak_cuda_memory_bytes": PEAK_CUDA_MEMORY_BUDGET_BYTES,
            "global_energy_must_not_increase": True,
            "two_sweep_energy_consistency_tolerance": (
                SWEEP_CONSISTENCY_TOLERANCE
            ),
        },
        "source_state": source_record,
        "configuration": {
            "particles": 8,
            "basis_order": 10,
            "scale": scale,
            "scale_ratio": scale_ratio,
            "fourier_order": 96,
            "local_quadrature_order": 192,
            "mpo_maximum_bond": 128,
            "mps_maximum_bond": 32,
            "local_lanczos_maximum_iterations": (
                arguments.maximum_iterations
            ),
            "sweeps": arguments.sweeps,
        },
        "mpo_build": {
            "construction": build["construction"],
            "dense_raw_fourier_bulk_materialized": build[
                "dense_raw_fourier_bulk_materialized"
            ],
            "maximum_intermediate_tensor_elements": build[
                "maximum_intermediate_tensor_elements"
            ],
        },
        "heff_contraction": {
            "strategy": "four_explicit_two_operand_einsums",
            "maximum_staged_temporary_element_bound": (
                staged_temporary_element_bound
            ),
            "maximum_staged_temporary_float64_bytes": (
                8 * staged_temporary_element_bound
            ),
            "pre_change_observed_failed_allocation": "78.12 GiB",
        },
        "initial_energy": initial_energy,
        "history": result["history"],
        "final_energy": result["final_energy"],
        "energy_change": result["final_energy"] - initial_energy,
        "two_sweep_energy_absolute_difference": sweep_consistency,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": peak_memory,
    }
    record["memory_budget_pass"] = (
        peak_memory < PEAK_CUDA_MEMORY_BUDGET_BYTES
    )
    record["energy_nonincrease_pass"] = (
        result["final_energy"] <= initial_energy + 1e-10
    )
    record["sweep_consistency_pass"] = (
        sweep_consistency is not None
        and sweep_consistency < SWEEP_CONSISTENCY_TOLERANCE
    )
    record["chi32_local_resource_audit_pass"] = (
        record["memory_budget_pass"]
        and record["energy_nonincrease_pass"]
        and record["sweep_consistency_pass"]
        and not build["dense_raw_fourier_bulk_materialized"]
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0 if record["chi32_local_resource_audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
