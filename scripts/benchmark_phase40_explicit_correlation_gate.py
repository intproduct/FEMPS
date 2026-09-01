"""Run the preregistered Phase 40 explicit-correlation differentiator gate.

The production axes, seeds, budgets, and acceptance rule are frozen in
``docs/exec-plans/active/phase40.md``.  This runner intentionally materializes
an ``N=2`` product quadrature grid as a bounded truth/gradient oracle.  It is
not a scalable many-particle contraction claim and it does not create Paper B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import torch

from femps.algorithms import (
    CorrelatedExteriorConfig,
    DiagonalPathConfig,
    canonical_slater_orbitals,
    canonical_two_orbital_carrier,
    correlated_two_fermion_observables,
    load_diagonal_path_checkpoint,
    run_correlated_exterior_optimization,
    run_diagonal_path_variable_projection,
    solve_generalized_hermitian,
)
from femps.basis import harmonic_hamiltonian
from femps.exterior import diagonal_path_hamiltonian_matrices
from femps.hamiltonians import (
    antisymmetric_two_particle_hamiltonian,
    soft_coulomb_operator,
    soft_coulomb_two_fermion_relative_grid_energy,
)


DEFAULT_OUTPUT = Path("docs/experiments/results/phase40_explicit_correlation_gate.json")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/phase40_explicit_correlation_gate")
D_AXIS = (2, 4, 6, 8, 10, 12)
P_FEATURES: dict[int, tuple[float, ...]] = {
    0: (),
    1: (1.0,),
    3: (0.25, 1.0, 4.0),
    5: (0.0625, 0.25, 1.0, 4.0, 16.0),
}
K_AXIS = (1, 2, 4)
CORRELATED_SEEDS = (40001, 40002, 40003)
NOCI_SEEDS = (40101, 40102, 40103)
OPTIMIZATION_QUADRATURE = 96
AUDIT_QUADRATURE = (128, 160)
ENERGY_QUADRATURE_TOLERANCE = 1e-7
NORM_QUADRATURE_TOLERANCE = 1e-8
ANTISYMMETRY_TOLERANCE = 1e-12
GRADIENT_TOLERANCE = 1e-6
NORM_TOLERANCE = 1e-10
EXPLICIT_CONTRACTION_TOLERANCE = 1e-10


def _sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _real_tensor(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def _complex_tensor(value: torch.Tensor) -> dict[str, list[Any]]:
    detached = value.detach().cpu()
    return {"real": detached.real.tolist(), "imag": detached.imag.tolist()}


def _operators(dimension: int):
    one_body = harmonic_hamiltonian(dimension, dtype=torch.complex128)
    interaction, diagnostics = soft_coulomb_operator(
        dimension,
        quadrature_order=128,
        coupling=1.0,
        softening=1.0,
        relative_threshold=0.0,
        factorization_backend="physical",
        dtype=torch.complex128,
        device="cpu",
    )
    return one_body, interaction, diagnostics


def _gradient_check(
    raw_carrier: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
) -> dict[str, Any]:
    raw = raw_carrier.detach().clone().requires_grad_(True)
    amp = amplitudes.detach().clone().requires_grad_(True)

    def energy(raw_value: torch.Tensor, amplitude_value: torch.Tensor) -> torch.Tensor:
        return correlated_two_fermion_observables(
            canonical_two_orbital_carrier(raw_value),
            amplitude_value,
            exponents,
            quadrature_order=OPTIMIZATION_QUADRATURE,
            coupling=1.0,
            softening=1.0,
        ).energy

    value = energy(raw, amp)
    gradient_raw, gradient_amp = torch.autograd.grad(
        value, (raw, amp), allow_unused=True
    )
    row = 2 if raw.shape[0] > 2 else 0
    column = 0
    step = 1e-5
    plus_raw = raw.detach().clone()
    minus_raw = raw.detach().clone()
    plus_raw[row, column] += step
    minus_raw[row, column] -= step
    raw_fd = (
        energy(plus_raw, amp.detach()) - energy(minus_raw, amp.detach())
    ) / (2.0 * step)
    raw_ad = gradient_raw[row, column] if gradient_raw is not None else raw_fd.new_zeros(())
    result: dict[str, Any] = {
        "step": step,
        "carrier_index": [row, column],
        "carrier_autodiff": float(raw_ad),
        "carrier_central_finite_difference": float(raw_fd),
        "carrier_absolute_difference": float(torch.abs(raw_ad - raw_fd)),
        "correlator": None,
    }
    if amp.numel():
        plus_amp = amp.detach().clone()
        minus_amp = amp.detach().clone()
        plus_amp[0] += step
        minus_amp[0] -= step
        amp_fd = (
            energy(raw.detach(), plus_amp) - energy(raw.detach(), minus_amp)
        ) / (2.0 * step)
        amp_ad = gradient_amp[0] if gradient_amp is not None else amp_fd.new_zeros(())
        result["correlator"] = {
            "index": 0,
            "autodiff": float(amp_ad),
            "central_finite_difference": float(amp_fd),
            "absolute_difference": float(torch.abs(amp_ad - amp_fd)),
        }
    return result


def _correlated_point(
    dimension: int,
    feature_count: int,
    seed: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    config = CorrelatedExteriorConfig(
        basis_order=dimension,
        exponents=P_FEATURES[feature_count],
        seed=seed,
        quadrature_order=OPTIMIZATION_QUADRATURE,
        adam_steps=200,
        adam_learning_rate=0.03,
        lbfgs_steps=80,
        lbfgs_learning_rate=0.5,
        record_points=10,
        orbital_noise_scale=1e-3,
    )
    checkpoint = checkpoint_dir / f"correlated_D{dimension}_P{feature_count}_seed{seed}.pt"
    optimized = run_correlated_exterior_optimization(config, checkpoint_path=checkpoint)
    orbitals = optimized["orbitals"]
    raw_carrier = optimized["raw_carrier"]
    amplitudes = optimized["amplitudes"]
    assert isinstance(orbitals, torch.Tensor)
    assert isinstance(raw_carrier, torch.Tensor)
    assert isinstance(amplitudes, torch.Tensor)
    exponents = torch.tensor(P_FEATURES[feature_count], dtype=torch.float64)
    audits = {
        str(order): correlated_two_fermion_observables(
            orbitals,
            amplitudes,
            exponents,
            quadrature_order=order,
            coupling=1.0,
            softening=1.0,
        )
        for order in AUDIT_QUADRATURE
    }
    q128 = audits["128"]
    q160 = audits["160"]
    energy_change = abs(float(q160.energy) - float(q128.energy))
    norm_change = float(torch.abs(q160.norm - q128.norm) / torch.abs(q160.norm))
    gradient = _gradient_check(raw_carrier, amplitudes, exponents)
    correlator_gradient_error = (
        0.0
        if gradient["correlator"] is None
        else gradient["correlator"]["absolute_difference"]
    )
    validation = {
        "antisymmetry_pass": float(q160.antisymmetry_residual)
        <= ANTISYMMETRY_TOLERANCE,
        "correlator_symmetry_pass": float(q160.correlator_symmetry_residual)
        <= ANTISYMMETRY_TOLERANCE,
        "energy_quadrature_pass": energy_change <= ENERGY_QUADRATURE_TOLERANCE,
        "norm_quadrature_pass": norm_change <= NORM_QUADRATURE_TOLERANCE,
        "carrier_gradient_pass": gradient["carrier_absolute_difference"]
        <= GRADIENT_TOLERANCE,
        "correlator_gradient_pass": correlator_gradient_error <= GRADIENT_TOLERANCE,
    }
    validation["all_pass"] = all(validation.values())
    return {
        "D": dimension,
        "P": feature_count,
        "seed": seed,
        "method": "symmetric_explicit_correlator_times_chi1_exterior_carrier",
        "initialization": "canonical occupied orbitals plus seeded 1e-3 virtual noise; zero correlator amplitudes",
        "orbitals": _real_tensor(orbitals),
        "raw_carrier": _real_tensor(raw_carrier),
        "amplitudes": _real_tensor(amplitudes),
        "exponents": list(P_FEATURES[feature_count]),
        "energy": float(q160.energy),
        "energy_variance": float(q160.energy_variance),
        "raw_norm": float(q160.norm),
        "antisymmetry_residual": float(q160.antisymmetry_residual),
        "correlator_symmetry_residual": float(q160.correlator_symmetry_residual),
        "energy_uncertainty_q128_q160": energy_change,
        "relative_norm_change_q128_q160": norm_change,
        "quadrature_audit": {
            order: {
                "energy": float(result.energy),
                "norm": float(result.norm),
                "energy_variance": float(result.energy_variance),
                "materialized_coordinate_values": result.materialized_coordinate_values,
            }
            for order, result in audits.items()
        },
        "gradient_check": gradient,
        "validation": validation,
        "optimized_real_parameter_count": optimized["optimized_real_parameter_count"],
        "optimizer": {
            "trace": optimized["trace"],
            "lbfgs_closure_calls": optimized["lbfgs_closure_calls"],
        },
        "elapsed_seconds": optimized["elapsed_seconds"],
        "peak_cpu_rss_bytes": optimized["peak_cpu_rss_bytes"],
        "materialization": {
            "bounded_Q_squared_grid": True,
            "largest_coordinate_grid_values": 160**2,
            "full_alternating_coefficient_tensor": False,
            "virtual_paths": 0,
        },
        "checkpoint": str(checkpoint),
    }


def _noci_point(
    dimension: int,
    terms: int,
    seed: int,
    checkpoint_dir: Path,
    operators,
) -> dict[str, Any]:
    one_body, interaction, diagnostics = operators
    config = DiagonalPathConfig(
        basis_order=dimension,
        particles=2,
        terms=terms,
        interaction_model="soft_coulomb",
        soft_coulomb_coupling=1.0,
        soft_coulomb_softening=1.0,
        soft_coulomb_quadrature_order=128,
        soft_coulomb_relative_threshold=0.0,
        steps=200,
        learning_rate=0.01,
        final_learning_rate=1e-5,
        seed=seed,
        device="cpu",
        record_points=10,
        checkpoint_every=200,
        lbfgs_refinement_steps=80,
        lbfgs_learning_rate=0.5,
        truth_maximum_dimension=math.comb(dimension, 2),
        particle_tensor_maximum_coefficients=dimension**2,
    )
    checkpoint = checkpoint_dir / f"noci_D{dimension}_K{terms}_seed{seed}.pt"
    result = run_diagonal_path_variable_projection(
        config,
        checkpoint_path=checkpoint,
        operators=(one_body, interaction),
        operator_id=f"phase40_soft_coulomb_D{dimension}_Q128_exact_physical",
    )
    payload = load_diagonal_path_checkpoint(checkpoint)
    raw = payload["best_raw"]
    orbitals = canonical_slater_orbitals(raw)
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        orbitals,
        one_body,
        two_body_left=interaction.left,
        two_body_right=interaction.right,
        two_body_weights=interaction.weights,
    )
    solved = solve_generalized_hermitian(
        hamiltonian, overlap, relative_threshold=config.overlap_relative_threshold
    )
    validation = {
        "completed": result["completed"],
        "norm_pass": result["norm_error"] <= NORM_TOLERANCE,
        "structural_antisymmetry_pass": result["structural_antisymmetry_residual"]
        <= ANTISYMMETRY_TOLERANCE,
        "materialized_antisymmetry_pass": result["materialized_antisymmetry_residual"]
        <= ANTISYMMETRY_TOLERANCE,
        "explicit_contraction_pass": result["polynomial_explicit_absolute_difference"]
        <= EXPLICIT_CONTRACTION_TOLERANCE,
        "no_virtual_path_enumeration": result["structural_counts"][
            "enumerated_virtual_paths"
        ]
        == 0,
    }
    validation["all_pass"] = all(validation.values())
    return {
        "D": dimension,
        "K": terms,
        "seed": seed,
        "method": "optimized_fixed_K_NOCI_control",
        "orbitals": _complex_tensor(orbitals),
        "linear_amplitudes": _complex_tensor(solved.amplitudes),
        "energy": result["energy"],
        "finite_basis_ci_energy": result["finite_basis_reference_energy"],
        "energy_variance": result["energy_variance"],
        "norm": result["norm"],
        "norm_error": result["norm_error"],
        "structural_antisymmetry_residual": result[
            "structural_antisymmetry_residual"
        ],
        "materialized_antisymmetry_residual": result[
            "materialized_antisymmetry_residual"
        ],
        "polynomial_explicit_absolute_difference": result[
            "polynomial_explicit_absolute_difference"
        ],
        "retained_rank": result["retained_rank"],
        "optimized_real_parameter_count": 2 * terms * dimension * 2,
        "linear_solved_real_parameter_count": 2 * terms,
        "optimizer": {
            "history": result["history"],
            "refinement": result["refinement"],
        },
        "elapsed_seconds": result["total_elapsed_seconds_this_call"],
        "peak_cpu_rss_bytes": result["peak_cpu_rss_bytes"],
        "operator": {
            "factor_rank": diagnostics.retained_rank,
            "dense_relative_factorization_error": diagnostics.dense_relative_factorization_error,
        },
        "materialization": {
            "bounded_Q_squared_grid": False,
            "full_exterior_vector_for_truth": True,
            "full_particle_tensor_for_antisymmetry_audit": True,
            "virtual_paths": 0,
        },
        "validation": validation,
        "checkpoint": str(checkpoint),
    }


def _best_lowest_energy(points: list[dict[str, Any]]) -> dict[str, Any]:
    return min(points, key=lambda point: point["energy"])


def _finalize(record: dict[str, Any]) -> dict[str, Any]:
    reference = record["reference"]["continuum_grid_energy"]
    correlated = record["correlated_points"]
    noci = record["noci_points"]
    same_basis_ci = record["same_basis_ci"]
    summaries = []
    for dimension in D_AXIS:
        corr_seeds = [
            point
            for point in correlated
            if point["D"] == dimension and point["P"] == 5
        ]
        noci_seeds = [
            point for point in noci if point["D"] == dimension and point["K"] == 4
        ]
        best_corr = _best_lowest_energy(corr_seeds)
        best_noci = _best_lowest_energy(noci_seeds)
        noci_error = abs(best_noci["energy"] - reference)
        seed_pass = [
            abs(point["energy"] - reference)
            + point["energy_uncertainty_q128_q160"]
            <= 0.5 * noci_error
            and point["validation"]["all_pass"]
            for point in corr_seeds
        ]
        best_corr_error = abs(best_corr["energy"] - reference)
        error_advantage = (
            best_corr_error + best_corr["energy_uncertainty_q128_q160"]
            <= 0.5 * noci_error
        )
        parameter_advantage = (
            best_corr["optimized_real_parameter_count"]
            <= best_noci["optimized_real_parameter_count"]
        )
        validation_pass = best_corr["validation"]["all_pass"] and best_noci[
            "validation"
        ]["all_pass"]
        reproducible = sum(seed_pass) >= 2
        point_pass = (
            error_advantage
            and parameter_advantage
            and validation_pass
            and reproducible
        )
        ci_energy = next(point["energy"] for point in same_basis_ci if point["D"] == dimension)
        summaries.append(
            {
                "D": dimension,
                "best_P5_seed": best_corr["seed"],
                "best_P5_energy": best_corr["energy"],
                "best_P5_reference_error": best_corr_error,
                "best_P5_uncertainty": best_corr["energy_uncertainty_q128_q160"],
                "best_K4_seed": best_noci["seed"],
                "best_K4_energy": best_noci["energy"],
                "best_K4_reference_error": noci_error,
                "same_basis_ci_energy": ci_energy,
                "same_basis_ci_reference_error": abs(ci_energy - reference),
                "P5_to_K4_error_ratio": (
                    best_corr_error / noci_error if noci_error else None
                ),
                "P5_real_parameters": best_corr["optimized_real_parameter_count"],
                "K4_real_nonlinear_parameters": best_noci[
                    "optimized_real_parameter_count"
                ],
                "P5_elapsed_seconds": best_corr["elapsed_seconds"],
                "K4_elapsed_seconds": best_noci["elapsed_seconds"],
                "P5_seed_passes": seed_pass,
                "error_advantage": error_advantage,
                "parameter_envelope_pass": parameter_advantage,
                "validation_pass": validation_pass,
                "reproducible_at_least_two_seeds": reproducible,
                "point_pass": point_pass,
            }
        )
    consecutive = [
        [left["D"], right["D"]]
        for left, right in zip(summaries, summaries[1:])
        if left["point_pass"] and right["point_pass"]
    ]
    all_correlated_valid = all(
        point["validation"]["all_pass"] for point in correlated
    )
    all_noci_valid = all(point["validation"]["all_pass"] for point in noci)
    record["comparison"] = {
        "primary": "P5 explicit correlation versus K4 NOCI at fixed D",
        "selection_rule": "lowest variational energy across the three frozen seeds",
        "same_basis_disclosure": "the correlated state is not confined to Lambda^2 V_D; CI is a comparator, not an equal-space bound",
        "D_axis": summaries,
        "consecutive_advantage_pairs": consecutive,
    }
    record["acceptance"] = {
        "all_correlated_validation_pass": all_correlated_valid,
        "all_noci_validation_pass": all_noci_valid,
        "two_consecutive_D_advantages": bool(consecutive),
        "phase40_differentiator_pass": (
            all_correlated_valid and all_noci_valid and bool(consecutive)
        ),
        "publication_consequence": "no Paper B; independent reproduction is still required even if this gate passes",
    }
    record["status"] = "complete"
    return record


def run(output: Path, checkpoint_dir: Path, *, resume: bool) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(40000)
    torch.use_deterministic_algorithms(True)
    if resume and output.exists():
        record = json.loads(output.read_text(encoding="utf-8"))
        if record.get("schema_version") != 1:
            raise ValueError("cannot resume an incompatible Phase 40 artifact")
    else:
        record = {
            "schema_version": 1,
            "experiment": "phase40_n2_explicit_correlation_differentiator_gate",
            "evidence_level": "preregistered numerical evidence",
            "scientific_boundary": "bounded N=2 Q^2 truth oracle and fixed-K NOCI controls; not a scalable contraction result, generic FEMPS advantage, or Paper B",
            "status": "in_progress",
            "frozen_config": {
                "D_axis": list(D_AXIS),
                "P_features": {str(key): list(value) for key, value in P_FEATURES.items()},
                "K_axis": list(K_AXIS),
                "correlated_seeds": list(CORRELATED_SEEDS),
                "noci_seeds": list(NOCI_SEEDS),
                "optimization_quadrature": OPTIMIZATION_QUADRATURE,
                "audit_quadrature": list(AUDIT_QUADRATURE),
                "correlated_optimizer": {
                    "adam_steps": 200,
                    "adam_learning_rate": 0.03,
                    "lbfgs_steps": 80,
                    "lbfgs_learning_rate": 0.5,
                },
                "noci_optimizer": {
                    "steps": 200,
                    "learning_rate": 0.01,
                    "final_learning_rate": 1e-5,
                    "lbfgs_steps": 80,
                    "lbfgs_learning_rate": 0.5,
                },
                "thresholds": {
                    "energy_q128_q160": ENERGY_QUADRATURE_TOLERANCE,
                    "relative_norm_q128_q160": NORM_QUADRATURE_TOLERANCE,
                    "antisymmetry": ANTISYMMETRY_TOLERANCE,
                    "gradient_absolute_difference": GRADIENT_TOLERANCE,
                    "norm": NORM_TOLERANCE,
                    "explicit_contraction": EXPLICIT_CONTRACTION_TOLERANCE,
                    "primary_error_ratio": 0.5,
                    "minimum_passing_seeds": 2,
                    "minimum_consecutive_D": 2,
                },
            },
            "reference": {
                "method": "independent relative-coordinate finite-difference grid",
                "intervals": 2000,
                "half_width": 10.0,
                "continuum_grid_energy": soft_coulomb_two_fermion_relative_grid_energy(
                    intervals=2000,
                    half_width=10.0,
                    coupling=1.0,
                    softening=1.0,
                ),
            },
            "correlated_points": [],
            "noci_points": [],
            "same_basis_ci": [],
            "source_hashes": {
                str(path): _sha256(path)
                for path in (
                    Path("src/femps/algorithms/correlated_exterior.py"),
                    Path("scripts/benchmark_phase40_explicit_correlation_gate.py"),
                    Path("docs/exec-plans/active/phase40.md"),
                    Path("docs/decisions/0030-preregister-n2-explicit-correlation-gate.md"),
                )
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "cuda_available_but_not_used": torch.cuda.is_available(),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            },
        }
        _write(output, record)

    correlated_keys = {
        (point["D"], point["P"], point["seed"])
        for point in record["correlated_points"]
    }
    for dimension in D_AXIS:
        for feature_count in P_FEATURES:
            for seed in CORRELATED_SEEDS:
                key = (dimension, feature_count, seed)
                if key in correlated_keys:
                    continue
                point = _correlated_point(
                    dimension, feature_count, seed, checkpoint_dir
                )
                point["reference_error"] = abs(
                    point["energy"] - record["reference"]["continuum_grid_energy"]
                )
                record["correlated_points"].append(point)
                _write(output, record)
                print(
                    f"correlated D={dimension} P={feature_count} seed={seed} "
                    f"error={point['reference_error']:.6e}",
                    flush=True,
                )

    noci_keys = {
        (point["D"], point["K"], point["seed"]) for point in record["noci_points"]
    }
    existing_ci = {point["D"] for point in record["same_basis_ci"]}
    for dimension in D_AXIS:
        operators = _operators(dimension)
        one_body, interaction, diagnostics = operators
        if dimension not in existing_ci:
            hamiltonian = antisymmetric_two_particle_hamiltonian(
                one_body, interaction
            )
            record["same_basis_ci"].append(
                {
                    "D": dimension,
                    "energy": float(torch.linalg.eigvalsh(hamiltonian)[0].real),
                    "exterior_dimension": math.comb(dimension, 2),
                    "operator_factor_rank": diagnostics.retained_rank,
                    "operator_factorization_error": diagnostics.dense_relative_factorization_error,
                }
            )
            _write(output, record)
        for terms in K_AXIS:
            for seed in NOCI_SEEDS:
                key = (dimension, terms, seed)
                if key in noci_keys:
                    continue
                point = _noci_point(
                    dimension, terms, seed, checkpoint_dir, operators
                )
                point["reference_error"] = abs(
                    point["energy"] - record["reference"]["continuum_grid_energy"]
                )
                record["noci_points"].append(point)
                _write(output, record)
                print(
                    f"NOCI D={dimension} K={terms} seed={seed} "
                    f"error={point['reference_error']:.6e}",
                    flush=True,
                )
    record = _finalize(record)
    _write(output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.checkpoint_dir, resume=arguments.resume)
    print(json.dumps(result["acceptance"], indent=2))


if __name__ == "__main__":
    main()
