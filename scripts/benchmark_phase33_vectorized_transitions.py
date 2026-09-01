"""Benchmark Phase 33 vectorized transitions and matched CPU/Blackwell solves."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch

from femps.algorithms import (
    DiagonalPathConfig,
    canonical_slater_orbitals,
    load_diagonal_path_checkpoint,
    run_diagonal_path_variable_projection,
    solve_generalized_hermitian,
)
from femps.benchmarks import ProcessRSSMonitor
from femps.exterior import (
    diagonal_path_exterior_coefficients,
    diagonal_path_hamiltonian_matrices,
    particle_tt_ranks_exterior_coefficients,
)
from femps.hamiltonians import (
    FactorizedTwoBodyOperator,
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)


PARTICLES = 6
DIMENSION = 10
TERMS = 4
QUADRATURE = 128
KERNEL_SEED = 3301
OPTIMIZATION_SEED = 3304
STEPS = 160
LBFGS_STEPS = 80
CPU_RSS_CAP_BYTES = 2 * 1024**3
GPU_MEMORY_CAP_BYTES = 4 * 1024**3
TIME_CAP_SECONDS = 600.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _random_orbitals(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (TERMS, DIMENSION, PARTICLES)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64)
    return canonical_slater_orbitals(torch.complex(real, imaginary))


def _move_operators(
    one_body: torch.Tensor,
    interaction: FactorizedTwoBodyOperator,
    device: torch.device,
) -> tuple[torch.Tensor, FactorizedTwoBodyOperator]:
    return (
        one_body.to(device),
        FactorizedTwoBodyOperator(
            interaction.left.to(device),
            interaction.right.to(device),
            interaction.weights.to(device),
        ),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _kernel_once(
    orbitals: torch.Tensor,
    one_body: torch.Tensor,
    interaction: FactorizedTwoBodyOperator,
    *,
    algorithm: str,
    backward: bool,
) -> dict:
    state = orbitals.detach().clone().requires_grad_(backward)
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        state,
        one_body,
        two_body_left=interaction.left,
        two_body_right=interaction.right,
        two_body_weights=interaction.weights,
        transition_algorithm=algorithm,
    )
    value = overlap.abs().square().sum() + hamiltonian.abs().square().sum()
    gradient = None
    if backward:
        gradient = torch.autograd.grad(value, state)[0]
    return {
        "value": float(value.detach().cpu()),
        "overlap": overlap.detach().cpu(),
        "hamiltonian": hamiltonian.detach().cpu(),
        "gradient": gradient.detach().cpu() if gradient is not None else None,
    }


def _kernel_timings(
    orbitals: torch.Tensor,
    one_body: torch.Tensor,
    interaction: FactorizedTwoBodyOperator,
    *,
    algorithm: str,
    backward: bool,
    repeats: int,
    device: torch.device,
) -> dict:
    _kernel_once(
        orbitals,
        one_body,
        interaction,
        algorithm=algorithm,
        backward=backward,
    )
    _synchronize(device)
    samples = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with ProcessRSSMonitor() as monitor:
        for _ in range(repeats):
            started = time.perf_counter()
            _kernel_once(
                orbitals,
                one_body,
                interaction,
                algorithm=algorithm,
                backward=backward,
            )
            _synchronize(device)
            samples.append(time.perf_counter() - started)
    memory = monitor.record()
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "cpu_memory": memory.as_dict(),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)))


def _complex_vector(values: torch.Tensor) -> list[list[float]]:
    cpu = values.detach().to(dtype=torch.complex128, device="cpu")
    return [[float(value.real), float(value.imag)] for value in cpu]


def _evaluate_checkpoint(
    result: dict,
    checkpoint: Path,
    one_body: torch.Tensor,
    interaction: FactorizedTwoBodyOperator,
    dense_hamiltonian: torch.Tensor,
    direct_ci_energy: float,
) -> dict:
    payload = load_diagonal_path_checkpoint(checkpoint)
    orbitals = canonical_slater_orbitals(payload["best_raw"])
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        orbitals,
        one_body,
        two_body_left=interaction.left,
        two_body_right=interaction.right,
        two_body_weights=interaction.weights,
    )
    solved = solve_generalized_hermitian(
        hamiltonian, overlap, relative_threshold=1e-10
    )
    coefficients = diagonal_path_exterior_coefficients(
        orbitals, solved.amplitudes
    )
    norm = torch.vdot(coefficients, coefficients).real
    acted = dense_hamiltonian @ coefficients
    energy = (torch.vdot(coefficients, acted) / norm).real
    residual = acted - energy * coefficients
    variance = torch.vdot(residual, residual).real / norm
    ranks = particle_tt_ranks_exterior_coefficients(
        coefficients, DIMENSION, PARTICLES
    )
    result.update(
        {
            "dense_quadrature_energy": float(energy),
            "dense_quadrature_variance": float(variance),
            "dense_quadrature_norm_error": float(abs(norm - 1.0)),
            "error_vs_direct_ci": float(energy) - direct_ci_energy,
            "ordinary_particle_tt_ranks_compact": list(ranks),
            "raw_exterior_coefficients": _complex_vector(coefficients),
        }
    )
    return result


def _config(device: str) -> DiagonalPathConfig:
    return DiagonalPathConfig(
        basis_order=DIMENSION,
        particles=PARTICLES,
        terms=TERMS,
        interaction_model="soft_coulomb",
        soft_coulomb_coupling=1.0,
        soft_coulomb_softening=1.0,
        soft_coulomb_quadrature_order=QUADRATURE,
        steps=STEPS,
        learning_rate=1e-3,
        final_learning_rate=1e-5,
        seed=OPTIMIZATION_SEED,
        device=device,
        record_points=10,
        checkpoint_every=STEPS,
        lbfgs_refinement_steps=LBFGS_STEPS,
        truth_maximum_dimension=1,
        particle_tensor_maximum_coefficients=100_000,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", default="cuda:2")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--phase32-artifact",
        type=Path,
        default=Path("docs/experiments/results/phase32_n6_convergence.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase33_vectorized_transitions"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase33_vectorized_transitions.json"
        ),
    )
    args = parser.parse_args()
    if args.repeats < 3:
        raise ValueError("registered timing audit requires at least three repeats")
    gpu_device = torch.device(args.gpu_device)
    if gpu_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registered Blackwell benchmark requires CUDA")
    if "Blackwell" not in torch.cuda.get_device_name(gpu_device):
        raise RuntimeError("registered GPU device is not the Blackwell card")

    phase32 = json.loads(args.phase32_artifact.read_text(encoding="utf-8"))
    if not phase32["acceptance"]["phase32_convergence_pass"]:
        raise RuntimeError("Phase 32 source gate did not pass")
    d10_audit = next(point for point in phase32["basis_audits"] if point["D"] == 10)
    direct_ci_energy = d10_audit["direct_ci"]["energy"]

    one_body = harmonic_pair_hamiltonian(
        DIMENSION, kappa=0.0, dtype=torch.complex128, device="cpu"
    )[0]
    interaction, factorization = soft_coulomb_operator(
        DIMENSION,
        quadrature_order=QUADRATURE,
        coupling=1.0,
        softening=1.0,
        relative_threshold=1e-13,
        factorization_backend="physical",
        dtype=torch.complex128,
        device="cpu",
    )
    dense_pair = soft_coulomb_dense_quadrature(
        DIMENSION,
        quadrature_order=QUADRATURE,
        coupling=1.0,
        softening=1.0,
        dtype=torch.complex128,
        device="cpu",
    )
    dense_hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, PARTICLES, dense_pair
    )
    direct_values = torch.linalg.eigvalsh(dense_hamiltonian)
    reconstructed_ci_energy = float(direct_values[0].real)
    if abs(reconstructed_ci_energy - direct_ci_energy) > 1e-11:
        raise RuntimeError("Phase 32 direct-CI source did not reproduce")

    kernel_orbitals_cpu = _random_orbitals(KERNEL_SEED)
    one_body_gpu, interaction_gpu = _move_operators(
        one_body, interaction, gpu_device
    )
    kernel_orbitals_gpu = kernel_orbitals_cpu.to(gpu_device)
    reference = _kernel_once(
        kernel_orbitals_cpu,
        one_body,
        interaction,
        algorithm="reference",
        backward=True,
    )
    vectorized_cpu = _kernel_once(
        kernel_orbitals_cpu,
        one_body,
        interaction,
        algorithm="auto",
        backward=True,
    )
    vectorized_gpu = _kernel_once(
        kernel_orbitals_gpu,
        one_body_gpu,
        interaction_gpu,
        algorithm="auto",
        backward=True,
    )
    assert reference["gradient"] is not None
    assert vectorized_cpu["gradient"] is not None
    assert vectorized_gpu["gradient"] is not None
    parity = {
        "cpu_auto_vs_reference_overlap_max_abs": _max_difference(
            vectorized_cpu["overlap"], reference["overlap"]
        ),
        "cpu_auto_vs_reference_hamiltonian_max_abs": _max_difference(
            vectorized_cpu["hamiltonian"], reference["hamiltonian"]
        ),
        "cpu_auto_vs_reference_gradient_max_abs": _max_difference(
            vectorized_cpu["gradient"], reference["gradient"]
        ),
        "gpu_auto_vs_cpu_auto_overlap_max_abs": _max_difference(
            vectorized_gpu["overlap"], vectorized_cpu["overlap"]
        ),
        "gpu_auto_vs_cpu_auto_hamiltonian_max_abs": _max_difference(
            vectorized_gpu["hamiltonian"], vectorized_cpu["hamiltonian"]
        ),
        "gpu_auto_vs_cpu_auto_gradient_max_abs": _max_difference(
            vectorized_gpu["gradient"], vectorized_cpu["gradient"]
        ),
    }
    timings = {}
    for label, orbitals, h1, pair, algorithm, device in (
        (
            "cpu_reference",
            kernel_orbitals_cpu,
            one_body,
            interaction,
            "reference",
            torch.device("cpu"),
        ),
        (
            "cpu_vectorized",
            kernel_orbitals_cpu,
            one_body,
            interaction,
            "auto",
            torch.device("cpu"),
        ),
        (
            "blackwell_vectorized",
            kernel_orbitals_gpu,
            one_body_gpu,
            interaction_gpu,
            "auto",
            gpu_device,
        ),
    ):
        timings[label] = {
            mode: _kernel_timings(
                orbitals,
                h1,
                pair,
                algorithm=algorithm,
                backward=backward,
                repeats=args.repeats,
                device=device,
            )
            for mode, backward in (("forward", False), ("forward_backward", True))
        }
        print(
            label,
            timings[label]["forward_backward"]["median_seconds"],
            flush=True,
        )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_orbitals = _random_orbitals(OPTIMIZATION_SEED)
    production = {}
    for label, device in (("cpu", "cpu"), ("blackwell", args.gpu_device)):
        checkpoint = args.checkpoint_dir / f"N6_D10_K4_seed3304_{label}.pt"
        result = run_diagonal_path_variable_projection(
            _config(device),
            checkpoint_path=checkpoint,
            initial_orbitals=initial_orbitals,
            operators=(one_body, interaction),
            operator_id="soft_N6_D10_Q128_physical_svd_phase33_matched",
        )
        production[label] = _evaluate_checkpoint(
            result,
            checkpoint,
            one_body,
            interaction,
            dense_hamiltonian,
            direct_ci_energy,
        )
        print(
            label,
            production[label]["dense_quadrature_energy"],
            production[label]["total_elapsed_seconds_this_call"],
            flush=True,
        )

    thresholds = {
        "cpu_reference_value_max_abs": 1e-11,
        "cpu_reference_gradient_max_abs": 1e-9,
        "cpu_gpu_value_max_abs": 1e-10,
        "cpu_gpu_gradient_max_abs": 1e-9,
        "production_energy_difference": 1e-10,
        "production_condition_relative_difference": 1e-8,
        "diagnostic_phase29_direct_ci_error": 5e-4,
        "diagnostic_phase29_variance": 5e-3,
        "norm_error": 1e-10,
        "antisymmetry_residual": 1e-12,
        "retained_condition_number": 1e8,
        "wall_time_seconds": TIME_CAP_SECONDS,
        "peak_cpu_rss_bytes": CPU_RSS_CAP_BYTES,
        "peak_cuda_memory_bytes": GPU_MEMORY_CAP_BYTES,
    }
    cpu_reference_parity_pass = bool(
        parity["cpu_auto_vs_reference_overlap_max_abs"] <= 1e-11
        and parity["cpu_auto_vs_reference_hamiltonian_max_abs"] <= 1e-11
        and parity["cpu_auto_vs_reference_gradient_max_abs"] <= 1e-9
    )
    cpu_gpu_parity_pass = bool(
        parity["gpu_auto_vs_cpu_auto_overlap_max_abs"] <= 1e-10
        and parity["gpu_auto_vs_cpu_auto_hamiltonian_max_abs"] <= 1e-10
        and parity["gpu_auto_vs_cpu_auto_gradient_max_abs"] <= 1e-9
    )
    per_backend_state_pass = {}
    production_accuracy_qc = {}
    for label, result in production.items():
        cuda_memory = result["peak_cuda_memory_bytes"]
        backend_memory_pass = (
            result["peak_cpu_rss_bytes"] <= CPU_RSS_CAP_BYTES
            if label == "cpu"
            else cuda_memory is not None
            and cuda_memory <= GPU_MEMORY_CAP_BYTES
        )
        per_backend_state_pass[label] = bool(
            result["completed"]
            and result["dense_quadrature_norm_error"] <= 1e-10
            and result["structural_antisymmetry_residual"] <= 1e-12
            and result["retained_condition_number"] <= 1e8
            and result["structural_counts"]["enumerated_virtual_paths"] == 0
            and result["structural_counts"]["materialized_particle_coefficients"]
            == 0
            and result["total_elapsed_seconds_this_call"] <= TIME_CAP_SECONDS
            and backend_memory_pass
        )
        production_accuracy_qc[label] = bool(
            -1e-9 <= result["error_vs_direct_ci"] <= 5e-4
            and result["dense_quadrature_variance"] <= 5e-3
        )
    production_energy_difference = abs(
        production["cpu"]["dense_quadrature_energy"]
        - production["blackwell"]["dense_quadrature_energy"]
    )
    condition_relative_difference = abs(
        production["cpu"]["retained_condition_number"]
        - production["blackwell"]["retained_condition_number"]
    ) / max(abs(production["cpu"]["retained_condition_number"]), 1.0)
    matched_production_pass = bool(
        all(per_backend_state_pass.values())
        and production_energy_difference <= 1e-10
        and condition_relative_difference <= 1e-8
    )
    blackwell_faster = bool(
        production["blackwell"]["total_elapsed_seconds_this_call"]
        < production["cpu"]["total_elapsed_seconds_this_call"]
    )
    accepted = bool(
        cpu_reference_parity_pass
        and cpu_gpu_parity_pass
        and matched_production_pass
        and factorization.dense_relative_factorization_error <= 1e-11
    )
    artifact = {
        "schema_version": 1,
        "experiment": "phase33_vectorized_transition_backend_gate",
        "evidence_level": "numerical",
        "scientific_boundary": (
            "one matched N6,D10,K4 workload; backend evidence, not asymptotic scaling"
        ),
        "source": {
            "phase32_artifact": args.phase32_artifact.as_posix(),
            "phase32_sha256": _sha256(args.phase32_artifact),
        },
        "model": {
            "N": PARTICLES,
            "D": DIMENSION,
            "K": TERMS,
            "Q": QUADRATURE,
            "physical_operator_svd_rank": interaction.rank,
            "physical_operator_svd_relative_error": (
                factorization.dense_relative_factorization_error
            ),
        },
        "registered_config": {
            "kernel_seed": KERNEL_SEED,
            "optimization_seed": OPTIMIZATION_SEED,
            "steps": STEPS,
            "lbfgs_steps": LBFGS_STEPS,
            "learning_rate": 1e-3,
            "final_learning_rate": 1e-5,
            "timing_repeats": args.repeats,
            "gpu_device": args.gpu_device,
            "gpu_name": torch.cuda.get_device_name(gpu_device),
            "truth_state_initialization": False,
        },
        "kernel_parity": parity,
        "kernel_timings": timings,
        "kernel_speedups": {
            "cpu_reference_over_vectorized_forward_backward": (
                timings["cpu_reference"]["forward_backward"]["median_seconds"]
                / timings["cpu_vectorized"]["forward_backward"]["median_seconds"]
            ),
            "cpu_reference_over_blackwell_forward_backward": (
                timings["cpu_reference"]["forward_backward"]["median_seconds"]
                / timings["blackwell_vectorized"]["forward_backward"][
                    "median_seconds"
                ]
            ),
        },
        "production": production,
        "matched_production": {
            "absolute_energy_difference": production_energy_difference,
            "retained_condition_relative_difference": (
                condition_relative_difference
            ),
            "blackwell_over_cpu_time_ratio": (
                production["blackwell"]["total_elapsed_seconds_this_call"]
                / production["cpu"]["total_elapsed_seconds_this_call"]
            ),
            "blackwell_faster": blackwell_faster,
            "selected_backend": "blackwell" if blackwell_faster else "cpu",
            "blackwell_admitted": matched_production_pass,
            "phase29_accuracy_quality_control_pass": all(
                production_accuracy_qc.values()
            ),
            "quality_control_interpretation": (
                "seed 3304 is a backend-parity workload; Phase 32 remains the "
                "registered physics-convergence evidence"
            ),
        },
        "thresholds": thresholds,
        "acceptance": {
            "cpu_reference_parity_pass": cpu_reference_parity_pass,
            "cpu_gpu_parity_pass": cpu_gpu_parity_pass,
            "per_backend_state_pass": per_backend_state_pass,
            "production_accuracy_quality_control": production_accuracy_qc,
            "matched_production_pass": matched_production_pass,
            "factorization_pass": (
                factorization.dense_relative_factorization_error <= 1e-11
            ),
            "phase33_backend_gate_pass": accepted,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    _write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": accepted,
                "blackwell_faster": blackwell_faster,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
