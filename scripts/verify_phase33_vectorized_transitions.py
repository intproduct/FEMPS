"""Independently verify the Phase 33 vectorized-transition backend artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from femps.algorithms import canonical_slater_orbitals
from femps.exterior import (
    diagonal_path_hamiltonian_matrices,
    diagonal_path_structural_counts,
    particle_tt_ranks_exterior_coefficients,
)
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)


N = 6
D = 10
K = 4
Q = 128
KERNEL_SEED = 3301
OPTIMIZATION_SEED = 3304


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(observed: float, expected: float, label: str, atol: float = 2e-10) -> None:
    if not math.isclose(observed, expected, rel_tol=2e-12, abs_tol=atol):
        raise AssertionError(f"{label} is inconsistent: {observed} != {expected}")


def _random_orbitals(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (K, D, N)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64)
    return canonical_slater_orbitals(torch.complex(real, imaginary))


def _kernel(
    orbitals: torch.Tensor,
    one_body: torch.Tensor,
    interaction,
    algorithm: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = orbitals.detach().clone().requires_grad_(True)
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        state,
        one_body,
        two_body_left=interaction.left,
        two_body_right=interaction.right,
        two_body_weights=interaction.weights,
        transition_algorithm=algorithm,
    )
    value = overlap.abs().square().sum() + hamiltonian.abs().square().sum()
    gradient = torch.autograd.grad(value, state)[0]
    return overlap.detach(), hamiltonian.detach(), gradient.detach()


def _coefficients(raw: list[list[float]]) -> torch.Tensor:
    return torch.tensor(
        [complex(real, imaginary) for real, imaginary in raw],
        dtype=torch.complex128,
    )


def verify_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["schema_version"] != 1:
        raise AssertionError("unexpected schema")
    if artifact["experiment"] != "phase33_vectorized_transition_backend_gate":
        raise AssertionError("unexpected experiment")
    if artifact["evidence_level"] != "numerical":
        raise AssertionError("backend results must remain numerical evidence")
    if "not asymptotic scaling" not in artifact["scientific_boundary"]:
        raise AssertionError("scientific boundary was weakened")

    source = artifact["source"]
    phase32_path = Path(source["phase32_artifact"])
    if _sha256(phase32_path) != source["phase32_sha256"]:
        raise AssertionError("Phase 32 source artifact changed")
    phase32 = json.loads(phase32_path.read_text(encoding="utf-8"))
    if not phase32["acceptance"]["phase32_convergence_pass"]:
        raise AssertionError("Phase 32 source gate is not passing")

    if artifact["model"]["N"] != N or artifact["model"]["D"] != D:
        raise AssertionError("model N/D changed")
    if artifact["model"]["K"] != K or artifact["model"]["Q"] != Q:
        raise AssertionError("model K/Q changed")
    registered = artifact["registered_config"]
    if (
        registered["kernel_seed"] != KERNEL_SEED
        or registered["optimization_seed"] != OPTIMIZATION_SEED
        or registered["steps"] != 160
        or registered["lbfgs_steps"] != 80
        or registered["learning_rate"] != 1e-3
        or registered["final_learning_rate"] != 1e-5
        or registered["timing_repeats"] != 5
        or registered["gpu_device"] != "cuda:2"
        or "Blackwell" not in registered["gpu_name"]
        or registered["truth_state_initialization"]
    ):
        raise AssertionError("registered backend configuration changed")

    one_body = harmonic_pair_hamiltonian(
        D, kappa=0.0, dtype=torch.complex128, device="cpu"
    )[0]
    interaction, diagnostics = soft_coulomb_operator(
        D,
        quadrature_order=Q,
        coupling=1.0,
        softening=1.0,
        relative_threshold=1e-13,
        factorization_backend="physical",
        dtype=torch.complex128,
        device="cpu",
    )
    if artifact["model"]["physical_operator_svd_rank"] != interaction.rank:
        raise AssertionError("factor rank changed")
    _close(
        artifact["model"]["physical_operator_svd_relative_error"],
        diagnostics.dense_relative_factorization_error,
        "factorization error",
        atol=1e-16,
    )
    kernel_orbitals = _random_orbitals(KERNEL_SEED)
    reference = _kernel(kernel_orbitals, one_body, interaction, "reference")
    vectorized = _kernel(kernel_orbitals, one_body, interaction, "auto")
    recomputed_parity = {
        "cpu_auto_vs_reference_overlap_max_abs": float(
            torch.max(torch.abs(vectorized[0] - reference[0]))
        ),
        "cpu_auto_vs_reference_hamiltonian_max_abs": float(
            torch.max(torch.abs(vectorized[1] - reference[1]))
        ),
        "cpu_auto_vs_reference_gradient_max_abs": float(
            torch.max(torch.abs(vectorized[2] - reference[2]))
        ),
    }
    for label, expected in recomputed_parity.items():
        _close(artifact["kernel_parity"][label], expected, label, atol=1e-15)

    timings = artifact["kernel_timings"]
    for backend in ("cpu_reference", "cpu_vectorized", "blackwell_vectorized"):
        for mode in ("forward", "forward_backward"):
            record = timings[backend][mode]
            if len(record["samples_seconds"]) != 5:
                raise AssertionError("timing sample count changed")
            _close(
                record["median_seconds"],
                statistics.median(record["samples_seconds"]),
                "timing median",
                atol=1e-15,
            )
            if record["minimum_seconds"] != min(record["samples_seconds"]):
                raise AssertionError("timing minimum is inconsistent")
    cpu_speedup = (
        timings["cpu_reference"]["forward_backward"]["median_seconds"]
        / timings["cpu_vectorized"]["forward_backward"]["median_seconds"]
    )
    gpu_speedup = (
        timings["cpu_reference"]["forward_backward"]["median_seconds"]
        / timings["blackwell_vectorized"]["forward_backward"]["median_seconds"]
    )
    _close(
        artifact["kernel_speedups"][
            "cpu_reference_over_vectorized_forward_backward"
        ],
        cpu_speedup,
        "CPU kernel speedup",
        atol=1e-14,
    )
    _close(
        artifact["kernel_speedups"][
            "cpu_reference_over_blackwell_forward_backward"
        ],
        gpu_speedup,
        "Blackwell kernel speedup",
        atol=1e-14,
    )

    dense_pair = soft_coulomb_dense_quadrature(
        D,
        quadrature_order=Q,
        coupling=1.0,
        softening=1.0,
        dtype=torch.complex128,
        device="cpu",
    )
    dense_hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
        one_body, N, dense_pair
    )
    phase32_d10 = next(
        point for point in phase32["basis_audits"] if point["D"] == D
    )
    direct_ci_energy = phase32_d10["direct_ci"]["energy"]
    expected_counts = diagonal_path_structural_counts(N, D, K, interaction.rank)
    reconstructed = {}
    for backend, result in artifact["production"].items():
        config = result["config"]
        expected_device = "cpu" if backend == "cpu" else "cuda:2"
        if (
            config["basis_order"] != D
            or config["particles"] != N
            or config["terms"] != K
            or config["steps"] != 160
            or config["lbfgs_refinement_steps"] != 80
            or config["seed"] != OPTIMIZATION_SEED
            or config["device"] != expected_device
        ):
            raise AssertionError("production configuration changed")
        if result["structural_counts"] != expected_counts:
            raise AssertionError("production structural counts changed")
        if result["structural_antisymmetry_residual"] != 0.0:
            raise AssertionError("structural antisymmetry residual changed")
        coefficients = _coefficients(result["raw_exterior_coefficients"])
        norm = torch.vdot(coefficients, coefficients).real
        acted = dense_hamiltonian @ coefficients
        energy = (torch.vdot(coefficients, acted) / norm).real
        residual = acted - energy * coefficients
        variance = torch.vdot(residual, residual).real / norm
        ranks = particle_tt_ranks_exterior_coefficients(coefficients, D, N)
        _close(result["dense_quadrature_energy"], float(energy), "production energy")
        _close(
            result["dense_quadrature_variance"],
            float(variance),
            "production variance",
        )
        _close(
            result["dense_quadrature_norm_error"],
            float(abs(norm - 1.0)),
            "production norm",
        )
        _close(
            result["error_vs_direct_ci"],
            float(energy) - direct_ci_energy,
            "production CI error",
        )
        if tuple(result["ordinary_particle_tt_ranks_compact"]) != ranks:
            raise AssertionError("production TT ranks changed")
        reconstructed[backend] = {
            "energy": float(energy),
            "variance": float(variance),
        }

    matched = artifact["matched_production"]
    energy_difference = abs(
        reconstructed["cpu"]["energy"] - reconstructed["blackwell"]["energy"]
    )
    _close(
        matched["absolute_energy_difference"],
        energy_difference,
        "matched energy difference",
        atol=1e-15,
    )
    condition_difference = abs(
        artifact["production"]["cpu"]["retained_condition_number"]
        - artifact["production"]["blackwell"]["retained_condition_number"]
    ) / max(
        abs(artifact["production"]["cpu"]["retained_condition_number"]), 1.0
    )
    _close(
        matched["retained_condition_relative_difference"],
        condition_difference,
        "matched condition difference",
        atol=1e-15,
    )
    time_ratio = (
        artifact["production"]["blackwell"]["total_elapsed_seconds_this_call"]
        / artifact["production"]["cpu"]["total_elapsed_seconds_this_call"]
    )
    _close(
        matched["blackwell_over_cpu_time_ratio"],
        time_ratio,
        "matched time ratio",
        atol=1e-14,
    )
    if matched["blackwell_faster"] != (time_ratio < 1.0):
        raise AssertionError("backend speed decision is inconsistent")
    if matched["selected_backend"] != "cpu" or not matched["blackwell_admitted"]:
        raise AssertionError("registered backend decision changed")
    if matched["phase29_accuracy_quality_control_pass"]:
        raise AssertionError("seed-3304 physics quality diagnostic must remain failed")

    thresholds = artifact["thresholds"]
    parity = artifact["kernel_parity"]
    cpu_reference_pass = bool(
        parity["cpu_auto_vs_reference_overlap_max_abs"]
        <= thresholds["cpu_reference_value_max_abs"]
        and parity["cpu_auto_vs_reference_hamiltonian_max_abs"]
        <= thresholds["cpu_reference_value_max_abs"]
        and parity["cpu_auto_vs_reference_gradient_max_abs"]
        <= thresholds["cpu_reference_gradient_max_abs"]
    )
    cpu_gpu_pass = bool(
        parity["gpu_auto_vs_cpu_auto_overlap_max_abs"]
        <= thresholds["cpu_gpu_value_max_abs"]
        and parity["gpu_auto_vs_cpu_auto_hamiltonian_max_abs"]
        <= thresholds["cpu_gpu_value_max_abs"]
        and parity["gpu_auto_vs_cpu_auto_gradient_max_abs"]
        <= thresholds["cpu_gpu_gradient_max_abs"]
    )
    per_backend = {}
    accuracy_qc = {}
    for backend, result in artifact["production"].items():
        memory_pass = (
            result["peak_cpu_rss_bytes"] <= thresholds["peak_cpu_rss_bytes"]
            if backend == "cpu"
            else result["peak_cuda_memory_bytes"]
            <= thresholds["peak_cuda_memory_bytes"]
        )
        per_backend[backend] = bool(
            result["completed"]
            and result["dense_quadrature_norm_error"] <= thresholds["norm_error"]
            and result["structural_antisymmetry_residual"]
            <= thresholds["antisymmetry_residual"]
            and result["retained_condition_number"]
            <= thresholds["retained_condition_number"]
            and result["total_elapsed_seconds_this_call"]
            <= thresholds["wall_time_seconds"]
            and memory_pass
        )
        accuracy_qc[backend] = bool(
            -1e-9
            <= result["error_vs_direct_ci"]
            <= thresholds["diagnostic_phase29_direct_ci_error"]
            and result["dense_quadrature_variance"]
            <= thresholds["diagnostic_phase29_variance"]
        )
    matched_pass = bool(
        all(per_backend.values())
        and energy_difference <= thresholds["production_energy_difference"]
        and condition_difference
        <= thresholds["production_condition_relative_difference"]
    )
    expected_acceptance = {
        "cpu_reference_parity_pass": cpu_reference_pass,
        "cpu_gpu_parity_pass": cpu_gpu_pass,
        "per_backend_state_pass": per_backend,
        "production_accuracy_quality_control": accuracy_qc,
        "matched_production_pass": matched_pass,
        "factorization_pass": (
            diagnostics.dense_relative_factorization_error <= 1e-11
        ),
    }
    expected_acceptance["phase33_backend_gate_pass"] = bool(
        cpu_reference_pass
        and cpu_gpu_pass
        and matched_pass
        and expected_acceptance["factorization_pass"]
    )
    if artifact["acceptance"] != expected_acceptance:
        raise AssertionError("acceptance record is inconsistent")
    if not expected_acceptance["phase33_backend_gate_pass"]:
        raise AssertionError("Phase 33 backend gate did not pass")
    return {
        "verified": True,
        "cpu_reference_speedup": cpu_speedup,
        "blackwell_reference_speedup": gpu_speedup,
        "selected_backend": matched["selected_backend"],
        "blackwell_admitted": matched["blackwell_admitted"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path(
            "docs/experiments/results/phase33_vectorized_transitions.json"
        ),
    )
    print(json.dumps(verify_artifact(parser.parse_args().artifact), indent=2))


if __name__ == "__main__":
    main()
