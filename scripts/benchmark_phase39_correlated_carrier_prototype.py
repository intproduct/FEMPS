"""Exploratory N=2 explicit-correlator exterior-carrier prototype.

This is a bounded materialization/AD audit, not a preregistered physics result
and not a production contraction algorithm.  It deliberately exposes its
``Q**2`` coordinate grid and dense same-basis CI controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from femps.algorithms.correlated_exterior import (
    correlated_two_fermion_observables,
    project_correlated_two_fermion_coefficients,
)
from femps.basis import harmonic_hamiltonian
from femps.benchmarks import ProcessRSSMonitor
from femps.hamiltonians import (
    antisymmetric_two_particle_hamiltonian,
    soft_coulomb_operator,
    soft_coulomb_two_fermion_relative_grid_energy,
)


DEFAULT_OUTPUT = Path(
    "docs/experiments/results/phase39_correlated_carrier_prototype.json"
)


def _canonical_carrier(order: int) -> torch.Tensor:
    orbitals = torch.zeros((order, 2), dtype=torch.float64)
    orbitals[0, 0] = 1.0
    orbitals[1, 1] = 1.0
    return orbitals


def _same_basis_ci_energy(order: int, quadrature_order: int) -> float:
    one_body = harmonic_hamiltonian(order, dtype=torch.complex128)
    interaction, _ = soft_coulomb_operator(
        order,
        quadrature_order=quadrature_order,
        coupling=1.0,
        softening=1.0,
        relative_threshold=0.0,
        factorization_backend="physical",
        dtype=torch.complex128,
    )
    hamiltonian = antisymmetric_two_particle_hamiltonian(one_body, interaction)
    return float(torch.linalg.eigvalsh(hamiltonian)[0].real)


def run(output: Path) -> dict[str, object]:
    torch.manual_seed(39001)
    torch.use_deterministic_algorithms(True)
    carrier = _canonical_carrier(4)
    exponents = torch.tensor(
        [0.0625, 0.25, 1.0, 4.0, 16.0], dtype=torch.float64
    )
    amplitudes = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    trace: list[dict[str, float | int]] = []
    start = time.perf_counter()
    with ProcessRSSMonitor() as monitor:
        optimizer = torch.optim.Adam([amplitudes], lr=0.03)
        for step in range(300):
            optimizer.zero_grad()
            result = correlated_two_fermion_observables(
                carrier,
                amplitudes,
                exponents,
                quadrature_order=96,
                coupling=1.0,
                softening=1.0,
            )
            result.energy.backward()
            optimizer.step()
            if step % 50 == 0 or step == 299:
                trace.append({"step": step, "energy": float(result.energy.detach())})

        optimizer_lbfgs = torch.optim.LBFGS(
            [amplitudes], lr=0.5, max_iter=100, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad()
            value = correlated_two_fermion_observables(
                carrier,
                amplitudes,
                exponents,
                quadrature_order=96,
                coupling=1.0,
                softening=1.0,
            ).energy
            value.backward()
            return value

        optimizer_lbfgs.step(closure)
        final_by_quadrature = {
            str(order): correlated_two_fermion_observables(
                carrier,
                amplitudes,
                exponents,
                quadrature_order=order,
                coupling=1.0,
                softening=1.0,
            )
            for order in (64, 96, 128, 160)
        }
        slater = correlated_two_fermion_observables(
            carrier,
            torch.zeros_like(amplitudes),
            exponents,
            quadrature_order=160,
            coupling=1.0,
            softening=1.0,
        )
        projection_axis = []
        for order in (4, 6, 8, 10, 12):
            coefficients = project_correlated_two_fermion_coefficients(
                carrier,
                amplitudes,
                exponents,
                projection_order=order,
                quadrature_order=160,
            ).detach()
            singular_values = torch.linalg.svdvals(coefficients)
            threshold = 1e-10 * singular_values[0]
            matrix_rank = int(torch.count_nonzero(singular_values > threshold))
            projection_axis.append(
                {
                    "projection_order": order,
                    "antisymmetric_matrix_rank": matrix_rank,
                    "slater_rank": matrix_rank // 2,
                    "relative_skew_residual": float(
                        torch.linalg.vector_norm(coefficients + coefficients.mT)
                        / torch.linalg.vector_norm(coefficients)
                    ),
                    "relative_rank_threshold": 1e-10,
                }
            )
        ci_axis = [
            {
                "basis_order": order,
                "same_basis_ci_energy": _same_basis_ci_energy(order, 128),
            }
            for order in (2, 4, 6, 8, 10, 12)
        ]
        continuum_reference = soft_coulomb_two_fermion_relative_grid_energy(
            intervals=2000,
            half_width=10.0,
            coupling=1.0,
            softening=1.0,
        )
        final = final_by_quadrature["160"]

        amplitudes_for_gradient = amplitudes.detach().clone().requires_grad_(True)
        gradient_energy = correlated_two_fermion_observables(
            carrier,
            amplitudes_for_gradient,
            exponents,
            quadrature_order=96,
            coupling=1.0,
            softening=1.0,
        ).energy
        gradient = torch.autograd.grad(gradient_energy, amplitudes_for_gradient)[0]
        difference_step = 1e-5
        plus = amplitudes.detach().clone()
        minus = amplitudes.detach().clone()
        plus[0] += difference_step
        minus[0] -= difference_step
        finite_difference = (
            correlated_two_fermion_observables(
                carrier,
                plus,
                exponents,
                quadrature_order=96,
                coupling=1.0,
                softening=1.0,
            ).energy
            - correlated_two_fermion_observables(
                carrier,
                minus,
                exponents,
                quadrature_order=96,
                coupling=1.0,
                softening=1.0,
            ).energy
        ) / (2.0 * difference_step)
    elapsed = time.perf_counter() - start

    q160 = final_by_quadrature["160"]
    record: dict[str, object] = {
        "schema_version": 1,
        "experiment": "phase39_correlated_exterior_n2_exploratory_prototype",
        "evidence_level": "exploratory numerical evidence",
        "scientific_boundary": (
            "bounded Q^2 materialization and dense-CI truth audit; not a "
            "preregistered benchmark, production contraction, VMC result, "
            "generic FEMPS claim, or method-paper gate pass"
        ),
        "model": {
            "particles": 2,
            "dimension": 1,
            "trap_omega": 1.0,
            "interaction": "soft_coulomb",
            "coupling": 1.0,
            "softening": 1.0,
            "carrier_basis": "unit-frequency harmonic functions",
            "carrier_basis_order": 4,
            "carrier": "canonical orbitals 0 and 1",
            "correlator": "exp(sum_m amplitude_m exp(-exponent_m (x1-x2)^2))",
            "correlator_exponents": exponents.tolist(),
        },
        "optimization": {
            "seed": 39001,
            "initialization": "zero correlator amplitudes",
            "adam_steps": 300,
            "adam_learning_rate": 0.03,
            "lbfgs_max_iterations": 100,
            "lbfgs_learning_rate": 0.5,
            "optimized_amplitudes": amplitudes.detach().tolist(),
            "trace": trace,
        },
        "final": {
            "energy": float(final.energy.detach()),
            "continuum_grid_reference_energy": continuum_reference,
            "absolute_reference_error": abs(
                float(final.energy.detach()) - continuum_reference
            ),
            "uncorrelated_slater_energy": float(slater.energy.detach()),
            "energy_improvement_over_same_carrier": float(
                slater.energy.detach() - final.energy.detach()
            ),
            "raw_norm": float(final.norm.detach()),
            "quadrature_norm_relative_change_q128_to_q160": float(
                (
                    torch.abs(final.norm - final_by_quadrature["128"].norm)
                    / torch.abs(final.norm)
                ).detach()
            ),
            "energy_variance": float(final.energy_variance.detach()),
            "antisymmetry_residual": float(final.antisymmetry_residual.detach()),
            "correlator_symmetry_residual": float(
                final.correlator_symmetry_residual.detach()
            ),
        },
        "quadrature_axis": [
            {
                "quadrature_order": int(order),
                "materialized_coordinate_values": result.materialized_coordinate_values,
                "energy": float(result.energy.detach()),
                "norm": float(result.norm.detach()),
                "energy_variance": float(result.energy_variance.detach()),
                "antisymmetry_residual": float(result.antisymmetry_residual.detach()),
            }
            for order, result in final_by_quadrature.items()
        ],
        "projection_rank_axis": projection_axis,
        "same_basis_ci_axis": ci_axis,
        "gradient_check": {
            "parameter": "amplitude[0]",
            "autodiff": float(gradient[0].detach()),
            "central_finite_difference": float(finite_difference.detach()),
            "absolute_difference": float(
                torch.abs(gradient[0] - finite_difference).detach()
            ),
            "step": difference_step,
        },
        "resources": {
            "elapsed_seconds": elapsed,
            "process_rss": monitor.record().as_dict(),
            "production_virtual_path_enumeration": 0,
            "bounded_coordinate_grid_materialized": True,
            "largest_materialized_coordinate_values": 160**2,
            "dense_ci_used_only_as_bounded_truth": True,
        },
        "interpretation": {
            "fixed_single_slater_projection_rank": 2,
            "correlated_projection_rank_grows_with_D": all(
                point["slater_rank"] > 1 for point in projection_axis
            ),
            "claim": (
                "The symmetric correlator preserves exact antisymmetry and "
                "produces a projected Slater rank greater than one. This is "
                "only a candidate differentiator; a preregistered matched-D "
                "comparison against optimized fixed-K NOCI is still required."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = run(arguments.output)
    print(json.dumps(result["final"], indent=2))


if __name__ == "__main__":
    main()
