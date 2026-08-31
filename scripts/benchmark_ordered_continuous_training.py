"""Formal multi-seed and CPU/GPU evidence for continuous ordered Gate D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from femps.algorithms.ordered_continuous_training import (
    OrderedContinuousTrainingConfig,
    random_uniform_functional_mps,
    train_ordered_continuous_mps,
)
from femps.baselines.ordered_continuous_mpo import (
    ordered_continuous_soft_coulomb_hamiltonian_mpo,
)
from femps.devices import resolve_device


def _run_training(config: OrderedContinuousTrainingConfig) -> dict[str, object]:
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    _, diagnostics = train_ordered_continuous_mps(config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "seed": config.seed,
        "initial_energy": diagnostics["initial_energy"],
        "final_energy": diagnostics["final_energy"],
        "energy_history": diagnostics["energy_history"],
        "max_bond": diagnostics["max_bond"],
        "mpo_max_bond": diagnostics["mpo_max_bond"],
        "mps_parameter_count": diagnostics["mps_parameter_count"],
        "physical_norm": diagnostics["physical_norm_after_projection"],
        "seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }


def _evaluate_parity(
    base_cores: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor], int | None]:
    from latticetn.mps import MPS

    cores = [core.to(device) for core in base_cores]
    mps = MPS.from_tensors(
        cores, dtype=torch.float64, device=device, requires_grad=True
    )
    mpo = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        3,
        4,
        5.0,
        8,
        interaction_quadrature_order=100,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    energy = mps.energy_with_MPO(mpo)
    energy.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gradients = [tensor.grad.detach().cpu() for tensor in mps.tensors]
    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    return energy.detach().cpu(), gradients, peak


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n2-seeds", nargs="+", type=int, default=[1601, 1602, 1603])
    parser.add_argument("--n4-seeds", nargs="+", type=int, default=[1680, 1681, 1682])
    parser.add_argument("--n2-steps", type=int, default=800)
    parser.add_argument("--n4-steps", type=int, default=1800)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase16_ordered_continuous_training.json"
        ),
    )
    arguments = parser.parse_args()
    device = resolve_device(arguments.device)
    if device.type != "cuda":
        raise RuntimeError("the formal record requires a CUDA device")

    n2_runs = [
        _run_training(
            OrderedContinuousTrainingConfig(
                particles=2,
                basis_order=12,
                distance_length=9.0,
                interaction_degree=20,
                interaction_quadrature_order=160,
                bond_dimension=12,
                steps=arguments.n2_steps,
                learning_rate=0.02,
                seed=seed,
                device=str(device),
            )
        )
        for seed in arguments.n2_seeds
    ]
    # The small Galerkin truth is constructed only after all blind N=2 runs.
    n2_hamiltonian = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        2, 12, 9.0, 20, interaction_quadrature_order=160
    ).to_dense()
    n2_truth = float(torch.linalg.eigvalsh(n2_hamiltonian)[0])
    for run in n2_runs:
        run["error_vs_post_training_galerkin_truth"] = run["final_energy"] - n2_truth

    n4_runs = [
        _run_training(
            OrderedContinuousTrainingConfig(
                particles=4,
                basis_order=10,
                distance_length=4.5,
                interaction_degree=20,
                interaction_quadrature_order=140,
                bond_dimension=32,
                steps=arguments.n4_steps,
                learning_rate=0.006,
                seed=seed,
                device=str(device),
            )
        )
        for seed in arguments.n4_seeds
    ]
    # This independent exterior-basis reference is read only after training.
    reference_path = Path(
        "docs/experiments/results/soft_coulomb_n4_truth_sweep.json"
    )
    reference_record = json.loads(reference_path.read_text(encoding="utf-8"))
    n4_reference = float(reference_record["basis_scan"][-1]["ground_energy"])
    for run in n4_runs:
        run["error_vs_post_training_exterior_D14_reference"] = (
            run["final_energy"] - n4_reference
        )

    parity_seed = random_uniform_functional_mps(
        3, 4, 4, seed=1690, dtype=torch.float64, device="cpu"
    )
    base_cores = [tensor.detach().clone() for tensor in parity_seed.tensors]
    cpu_energy, cpu_gradients, _ = _evaluate_parity(
        base_cores, torch.device("cpu")
    )
    gpu_energy, gpu_gradients, parity_peak = _evaluate_parity(base_cores, device)
    parity = {
        "energy_cpu": float(cpu_energy),
        "energy_gpu": float(gpu_energy),
        "energy_absolute_difference": float(torch.abs(cpu_energy - gpu_energy)),
        "gradient_max_absolute_difference": max(
            float(torch.max(torch.abs(cpu - gpu)))
            for cpu, gpu in zip(cpu_gradients, gpu_gradients, strict=True)
        ),
        "gpu_peak_memory_bytes": parity_peak,
    }

    n2_tolerance = 1e-5
    n4_reference_tolerance = 6e-3
    result = {
        "schema_version": 1,
        "experiment": "phase16_ordered_continuous_multiseed_training",
        "evidence_level": "blind_native_training_then_independent_truth_audit",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": "float64",
        "training_materializes_product_basis_state": False,
        "antisymmetry_mechanism": (
            "Dirichlet collision faces in one ordered chamber followed by "
            "exact signed permutation extension"
        ),
        "antisymmetry_residual": 0.0,
        "antisymmetry_residual_status": "exact_by_construction",
        "n2": {
            "configuration": {
                "basis": "dirichlet_sine",
                "basis_order": 12,
                "distance_length": 9.0,
                "interaction_degree": 20,
                "bond_dimension": 12,
                "steps": arguments.n2_steps,
            },
            "post_training_galerkin_truth": n2_truth,
            "declared_energy_tolerance": n2_tolerance,
            "runs": n2_runs,
            "all_seed_pass": all(
                abs(run["error_vs_post_training_galerkin_truth"]) < n2_tolerance
                for run in n2_runs
            ),
        },
        "n4": {
            "configuration": {
                "basis": "dirichlet_sine",
                "basis_order": 10,
                "distance_length": 4.5,
                "interaction_degree": 20,
                "bond_dimension": 32,
                "steps": arguments.n4_steps,
            },
            "post_training_reference": n4_reference,
            "reference_provenance": str(reference_path),
            "reference_is_numerical_not_continuum_bound": True,
            "declared_total_error_tolerance": n4_reference_tolerance,
            "runs": n4_runs,
            "all_seed_pass": all(
                abs(run["error_vs_post_training_exterior_D14_reference"])
                < n4_reference_tolerance
                for run in n4_runs
            ),
        },
        "cpu_gpu_parity": parity,
    }
    result["gate_record_pass"] = (
        result["n2"]["all_seed_pass"]
        and result["n4"]["all_seed_pass"]
        and parity["energy_absolute_difference"] < 1e-11
        and parity["gradient_max_absolute_difference"] < 1e-10
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate_record_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
