"""Blind N=6 GPU training followed by independent Phase 17 truth audits."""

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
    ordered_continuous_fourier_hamiltonian_mpo,
)
from femps.baselines.ordered_distance_mpo import compress_mpo
from femps.baselines.ordered_functional_mps import (
    particle_tensor_to_mps_tensors,
)
from femps.benchmarks.mpo_truth import (
    lowest_mpo_eigenpair,
    mpo_product_basis_matvec,
)
from femps.devices import resolve_device


N6_EXTERIOR_D12_REFERENCE = 25.049366416096817


def _configuration(
    *,
    scale: float,
    bond: int,
    steps: int,
    seed: int,
    device: torch.device,
) -> OrderedContinuousTrainingConfig:
    return OrderedContinuousTrainingConfig(
        particles=6,
        basis_order=8,
        distance_length=scale,
        distance_basis="odd_hermite",
        interaction_method="fourier_bessel",
        fourier_order=96,
        interaction_quadrature_order=160,
        mpo_max_bond=96,
        bond_dimension=bond,
        steps=steps,
        learning_rate=0.01,
        seed=seed,
        projection="tensor_norm",
        device=str(device),
    )


def _run_training(
    config: OrderedContinuousTrainingConfig,
    *,
    retain_state: bool = False,
):
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    mps, diagnostics = train_ordered_continuous_mps(config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    record = {
        "seed": config.seed,
        "scale": config.distance_length,
        "requested_mps_bond": config.bond_dimension,
        "steps": config.steps,
        "initial_energy": diagnostics["initial_energy"],
        "final_energy": diagnostics["final_energy"],
        "energy_history": diagnostics["energy_history"],
        "gradient_norm_history": diagnostics["grad_norm_history"],
        "state_norm_history": diagnostics["state_norm_history"],
        "physical_norm_after_projection": diagnostics[
            "physical_norm_after_projection"
        ],
        "canonical_residual": diagnostics["canonical_residual"],
        "actual_mps_maximum_bond": diagnostics["max_bond"],
        "mps_parameter_count": diagnostics["mps_parameter_count"],
        "raw_mpo_maximum_bond": diagnostics["uncompressed_mpo_max_bond"],
        "raw_mpo_tensor_elements": diagnostics[
            "uncompressed_mpo_tensor_elements"
        ],
        "compressed_mpo_maximum_bond": diagnostics["mpo_max_bond"],
        "compressed_mpo_tensor_elements": diagnostics["mpo_tensor_elements"],
        "mpo_compression_ranks": diagnostics["mpo_compression_ranks"],
        "mpo_local_discarded_norm_not_global_certificate": diagnostics[
            "mpo_compression_local_discarded_norm"
        ],
        "training_materializes_product_basis_state": diagnostics[
            "native_training_materializes_product_tensor"
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    return record, mps if retain_state else None


def _maximum_bond(mpo) -> int:
    return max(max(tensor.shape[:2]) for tensor in mpo.tensors)


def _n6_global_compression_audit(scale: float) -> dict[str, object]:
    raw = ordered_continuous_fourier_hamiltonian_mpo(
        6, 8, scale, 96, local_quadrature_order=160
    )
    generator = torch.Generator().manual_seed(1717)
    vector = torch.randn(8**6, generator=generator, dtype=torch.float64)
    reference = mpo_product_basis_matvec(raw, vector)
    reference_norm = torch.linalg.vector_norm(reference)
    points = []
    for maximum_bond in [64, 96, 128]:
        compressed, ranks, discarded = compress_mpo(raw, maximum_bond)
        difference = mpo_product_basis_matvec(compressed, vector) - reference
        points.append(
            {
                "requested_maximum_bond": maximum_bond,
                "retained_ranks": list(ranks),
                "local_discarded_norm_not_global_certificate": float(discarded),
                "global_random_action_relative_error": float(
                    torch.linalg.vector_norm(difference) / reference_norm
                ),
                "global_random_action_maximum_absolute_error": float(
                    torch.max(torch.abs(difference))
                ),
            }
        )
    return {
        "seed": 1717,
        "product_basis_dimension": 8**6,
        "dense_hamiltonian_materialized": False,
        "raw_maximum_bond": _maximum_bond(raw),
        "raw_tensor_elements": sum(tensor.numel() for tensor in raw.tensors),
        "points": points,
    }


def _galerkin_point(
    basis_order: int,
    scale: float,
    *,
    seed: int,
    initial_vector: torch.Tensor | None = None,
):
    raw = ordered_continuous_fourier_hamiltonian_mpo(
        6,
        basis_order,
        scale,
        96,
        local_quadrature_order=160,
    )
    compressed, ranks, discarded = compress_mpo(raw, 128)
    energy, vector, diagnostics = lowest_mpo_eigenpair(
        compressed,
        tolerance=5e-9,
        maximum_iterations=1000,
        seed=seed,
        initial_vector=initial_vector,
    )
    return (
        {
            "basis_order": basis_order,
            "scale": scale,
            "fourier_order": 96,
            "local_quadrature_order": 160,
            "raw_mpo_maximum_bond": _maximum_bond(raw),
            "truth_mpo_compression_bond": 128,
            "truth_mpo_compression_ranks": list(ranks),
            "truth_mpo_local_discarded_norm_not_global_certificate": float(
                discarded
            ),
            "ground_energy": energy,
            "error_vs_exterior_D12_reference": (
                energy - N6_EXTERIOR_D12_REFERENCE
            ),
            **diagnostics,
        },
        vector,
        compressed,
    )


def _tt_svd_capacity(
    exact_vector: torch.Tensor,
    truth_mpo,
    exact_energy: float,
) -> dict[str, object]:
    from latticetn.mps import MPS

    tensor = exact_vector.reshape((truth_mpo.dim,) * truth_mpo.length)
    points = []
    exact_norm = torch.vdot(exact_vector, exact_vector).real
    for bond in [1, 2, 4, 8, 16, 32, 64, 128, 512]:
        cores, ranks, discarded = particle_tensor_to_mps_tensors(
            tensor, max_bond=bond
        )
        compressed = MPS.from_tensors(cores, dtype=torch.float64)
        vector = compressed.to_dense().detach()
        vector_norm = torch.vdot(vector, vector).real
        fidelity = torch.vdot(exact_vector, vector).abs().square() / (
            exact_norm * vector_norm
        )
        energy = float(compressed.energy_with_MPO(truth_mpo).detach())
        points.append(
            {
                "requested_maximum_bond": bond,
                "retained_ranks": list(ranks),
                "energy": energy,
                "energy_error_vs_exact_galerkin_ground": energy - exact_energy,
                "fidelity_vs_exact_galerkin_ground": float(fidelity),
                "sequential_discarded_singular_value_norm": float(discarded),
            }
        )
    return {
        "method": "post-training TT-SVD of independent Lanczos ground state",
        "truth_state_materialization_is_post_training_audit_only": True,
        "points": points,
    }


def _trained_state_audit(
    retained_runs,
    exact_vector: torch.Tensor,
    exact_energy: float,
) -> list[dict[str, object]]:
    exact_norm = torch.vdot(exact_vector, exact_vector).real
    audits = []
    for record, mps in retained_runs:
        vector = mps.to_dense().detach().cpu()
        vector_norm = torch.vdot(vector, vector).real
        fidelity = torch.vdot(exact_vector, vector).abs().square() / (
            exact_norm * vector_norm
        )
        audits.append(
            {
                "seed": record["seed"],
                "energy_error_vs_exact_galerkin_ground": (
                    record["final_energy"] - exact_energy
                ),
                "fidelity_vs_exact_galerkin_ground": float(fidelity),
                "product_basis_state_materialized_after_training_only": True,
            }
        )
    return audits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scale-steps", type=int, default=300)
    parser.add_argument("--bond-steps", type=int, default=600)
    parser.add_argument("--seed-steps", type=int, default=800)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase17_unbounded_fourier_n6.json"
        ),
    )
    arguments = parser.parse_args()
    device = resolve_device(arguments.device)
    if device.type != "cuda":
        raise RuntimeError("the formal N=6 record requires a CUDA device")
    # Initialize the selected device before resetting its allocator statistics.
    gpu_name = torch.cuda.get_device_name(device)
    started = time.perf_counter()

    # All choices and optimization runs precede same-basis truth construction.
    scale_runs = []
    for scale in [0.55, 0.60, 0.65, 0.70, 0.75]:
        record, _ = _run_training(
            _configuration(
                scale=scale,
                bond=16,
                steps=arguments.scale_steps,
                seed=1710,
                device=device,
            )
        )
        scale_runs.append(record)
    selected_scale = min(
        scale_runs, key=lambda point: point["final_energy"]
    )["scale"]

    bond_runs = []
    for bond in [4, 8, 16, 32]:
        record, _ = _run_training(
            _configuration(
                scale=selected_scale,
                bond=bond,
                steps=arguments.bond_steps,
                seed=1721,
                device=device,
            )
        )
        bond_runs.append(record)

    retained_runs = []
    for seed in [1731, 1732, 1733]:
        record, mps = _run_training(
            _configuration(
                scale=selected_scale,
                bond=32,
                steps=arguments.seed_steps,
                seed=seed,
                device=device,
            ),
            retain_state=True,
        )
        retained_runs.append((record, mps))

    # Independent truth and capacity audits start only after every blind run.
    best_record, best_mps = min(
        retained_runs, key=lambda item: item[0]["final_energy"]
    )
    lanczos_initial = best_mps.to_dense().detach().cpu()
    n6_d8, exact_vector, truth_mpo = _galerkin_point(
        8,
        selected_scale,
        seed=1760,
        initial_vector=lanczos_initial,
    )
    exact_energy = n6_d8["ground_energy"]
    trained_audits = _trained_state_audit(
        retained_runs, exact_vector, exact_energy
    )
    for record, _ in retained_runs:
        audit = next(
            point for point in trained_audits if point["seed"] == record["seed"]
        )
        record.update(audit)

    basis_scan = []
    for basis_order in [4, 6]:
        point, _, _ = _galerkin_point(
            basis_order,
            selected_scale,
            seed=1770 + basis_order,
        )
        basis_scan.append(point)
    basis_scan.append(n6_d8)

    compression = _n6_global_compression_audit(selected_scale)
    tt_svd = _tt_svd_capacity(exact_vector, truth_mpo, exact_energy)
    n6_total_tolerance = 2e-2
    optimization_tolerance = 2e-3
    result = {
        "schema_version": 1,
        "experiment": "phase17_unbounded_fourier_n6_blind_training",
        "evidence_level": (
            "blind_native_global_AD_then_independent_Lanczos_and_TT_SVD"
        ),
        "device": str(device),
        "gpu": gpu_name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": "float64",
        "training_materializes_product_basis_state": False,
        "truth_is_constructed_after_all_training_runs": True,
        "configuration": {
            "particles": 6,
            "basis": "odd_hermite_half_line",
            "basis_order": 8,
            "selected_scale": selected_scale,
            "fourier_order": 96,
            "local_quadrature_order": 160,
            "training_mpo_maximum_bond": 96,
            "learning_rate": 0.01,
            "projection": "tensor_norm",
        },
        "blind_scale_scan": scale_runs,
        "blind_mps_bond_scan": bond_runs,
        "blind_multiseed_runs": [record for record, _ in retained_runs],
        "same_basis_galerkin_truth": n6_d8,
        "basis_order_scan_at_selected_scale": basis_scan,
        "post_training_tt_svd_capacity": tt_svd,
        "global_mpo_compression_audit": compression,
        "exterior_D12_numerical_reference": N6_EXTERIOR_D12_REFERENCE,
        "reference_is_numerical_not_continuum_bound": True,
        "declared_total_error_tolerance": n6_total_tolerance,
        "declared_optimization_error_tolerance": optimization_tolerance,
    }
    result["all_seed_optimization_pass"] = all(
        run["energy_error_vs_exact_galerkin_ground"] < optimization_tolerance
        for run, _ in retained_runs
    )
    result["basis_total_error_pass"] = (
        abs(exact_energy - N6_EXTERIOR_D12_REFERENCE) < n6_total_tolerance
    )
    training_compression = next(
        point
        for point in compression["points"]
        if point["requested_maximum_bond"] == 96
    )
    result["training_mpo_global_action_pass"] = (
        training_compression["global_random_action_relative_error"] < 1e-8
    )
    result["n6_controlled_point_pass"] = (
        result["all_seed_optimization_pass"]
        and result["basis_total_error_pass"]
        and result["training_mpo_global_action_pass"]
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["n6_controlled_point_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
