"""Phase 18 CPU controls for the multiscale basis and structured MPO."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from femps.baselines.ordered_continuous_fourier import (
    ordered_continuous_fourier_hamiltonian_compressed_mpo,
    ordered_continuous_fourier_hamiltonian_mpo,
)
from femps.baselines.ordered_distance_mpo import compress_mpo
from femps.basis.multiscale_odd_hermite import (
    multiscale_odd_hermite_basis_values,
    multiscale_odd_hermite_condition_number,
    multiscale_odd_hermite_derivative_matrix,
    multiscale_odd_hermite_negative_second_derivative_matrix,
    multiscale_odd_hermite_position_matrix,
    multiscale_odd_hermite_position_squared_matrix,
)
from femps.benchmarks.mpo_truth import (
    lowest_mpo_eigenpair,
    mpo_product_basis_matvec,
)


N2_CONTINUUM_REFERENCE = 2.553831733978763
N4_EXTERIOR_D14_REFERENCE = 11.023082853674637


def _maximum_bond(mpo) -> int:
    return max(max(tensor.shape[:2]) for tensor in mpo.tensors)


def _independent_basis_quadrature(
    order: int,
    scale: float,
    ratio: float,
    count: int = 700,
):
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(count)
    maximum_mode = (order + 1) // 2 - 1
    cutoff = scale * math.sqrt(ratio) * (
        math.sqrt(4 * (maximum_mode + 1) + 2) + 10
    )
    nodes = torch.from_numpy((raw_nodes + 1) * cutoff / 2)
    weights = torch.from_numpy(raw_weights * cutoff / 2)
    basis = multiscale_odd_hermite_basis_values(
        order, nodes, scale, ratio
    )
    return nodes, weights, basis


def _basis_analytic_controls() -> dict[str, object]:
    order = 5
    scale = 0.9
    ratio = 2.2
    nodes, weights, basis = _independent_basis_quadrature(
        order, scale, ratio
    )
    step = 2e-5
    plus = multiscale_odd_hermite_basis_values(
        order, nodes + step, scale, ratio
    )
    minus = multiscale_odd_hermite_basis_values(
        order, nodes - step, scale, ratio
    )
    derivative_values = (plus - minus) / (2 * step)
    quadrature_operators = {
        "overlap": torch.einsum("xm,x,xn->mn", basis, weights, basis),
        "derivative": torch.einsum(
            "xm,x,xn->mn", basis, weights, derivative_values
        ),
        "negative_second_derivative": torch.einsum(
            "xm,x,xn->mn", derivative_values, weights, derivative_values
        ),
        "position": torch.einsum(
            "xm,x,x,xn->mn", basis, weights, nodes, basis
        ),
        "position_squared": torch.einsum(
            "xm,x,x,xn->mn", basis, weights, nodes.square(), basis
        ),
    }
    analytic_operators = {
        "overlap": torch.eye(order, dtype=torch.float64),
        "derivative": multiscale_odd_hermite_derivative_matrix(
            order, scale, ratio
        ),
        "negative_second_derivative": (
            multiscale_odd_hermite_negative_second_derivative_matrix(
                order, scale, ratio
            )
        ),
        "position": multiscale_odd_hermite_position_matrix(
            order, scale, ratio
        ),
        "position_squared": (
            multiscale_odd_hermite_position_squared_matrix(
                order, scale, ratio
            )
        ),
    }
    residuals = {
        name: float(
            torch.max(
                torch.abs(analytic_operators[name] - quadrature_operator)
            )
        )
        for name, quadrature_operator in quadrature_operators.items()
    }
    boundary = multiscale_odd_hermite_basis_values(
        order, torch.zeros(1, dtype=torch.float64), scale, ratio
    )
    conditioning = [
        {
            "basis_order": basis_order,
            "scale_ratio": scale_ratio,
            "primitive_overlap_condition_number": (
                multiscale_odd_hermite_condition_number(
                    basis_order, 1.0, scale_ratio
                )
            ),
        }
        for basis_order in [4, 6, 8, 10, 12]
        for scale_ratio in [2.5, 3.0]
    ]
    return {
        "construction": (
            "symmetric Lowdin orthonormalization of two odd-Hermite scales"
        ),
        "all_primitives_obey_collision_dirichlet_boundary": True,
        "maximum_boundary_absolute_value": float(torch.max(torch.abs(boundary))),
        "independent_gauss_legendre_points": 700,
        "analytic_vs_independent_quadrature_maximum_absolute_residuals": (
            residuals
        ),
        "primitive_overlap_conditioning": conditioning,
    }


def _energy_point(
    particles: int,
    basis_order: int,
    scale: float,
    *,
    basis: str,
    ratio: float,
    seed: int,
) -> dict[str, object]:
    maximum_bond = max(64, basis_order**2)
    mpo, diagnostics = ordered_continuous_fourier_hamiltonian_compressed_mpo(
        particles,
        basis_order,
        scale,
        96,
        maximum_bond,
        distance_basis=basis,
        distance_scale_ratio=ratio,
        local_quadrature_order=160,
    )
    if particles == 2:
        energy = float(torch.linalg.eigvalsh(mpo.to_dense())[0])
        truth_diagnostics: dict[str, object] = {
            "truth_method": "dense_same_basis_eigensolve",
            "dense_hamiltonian_dimension": basis_order**2,
        }
    else:
        energy, _, lanczos = lowest_mpo_eigenpair(
            mpo,
            tolerance=2e-9,
            maximum_iterations=800,
            seed=seed,
        )
        truth_diagnostics = {
            "truth_method": "matrix_free_product_basis_lanczos",
            **lanczos,
        }
    reference = (
        N2_CONTINUUM_REFERENCE
        if particles == 2
        else N4_EXTERIOR_D14_REFERENCE
    )
    return {
        "basis": basis,
        "basis_order": basis_order,
        "scale": scale,
        "scale_ratio": ratio if basis == "multiscale_odd_hermite" else None,
        "ground_energy": energy,
        "error_vs_independent_numerical_reference": energy - reference,
        "compressed_mpo_maximum_bond": _maximum_bond(mpo),
        "mpo_construction": diagnostics["construction"],
        **truth_diagnostics,
    }


def _matched_basis_controls() -> dict[str, object]:
    results: dict[str, object] = {}
    for particles, orders, scales in [
        (2, [4, 6, 8, 10], [0.8, 0.9, 1.0, 1.1]),
        (4, [4, 6, 8], [0.6, 0.7, 0.8, 0.9, 1.0]),
    ]:
        order_records = []
        for basis_order in orders:
            odd_points = [
                _energy_point(
                    particles,
                    basis_order,
                    scale,
                    basis="odd_hermite",
                    ratio=2.0,
                    seed=1800 + 100 * particles + 10 * basis_order + index,
                )
                for index, scale in enumerate(scales)
            ]
            multiscale_points = [
                _energy_point(
                    particles,
                    basis_order,
                    scale,
                    basis="multiscale_odd_hermite",
                    ratio=ratio,
                    seed=(
                        1900
                        + 100 * particles
                        + 10 * basis_order
                        + 3 * scale_index
                        + ratio_index
                    ),
                )
                for scale_index, scale in enumerate(scales)
                for ratio_index, ratio in enumerate([2.0, 2.5, 3.0])
            ]
            odd_best = min(odd_points, key=lambda point: point["ground_energy"])
            multiscale_best = min(
                multiscale_points, key=lambda point: point["ground_energy"]
            )
            order_records.append(
                {
                    "basis_order": basis_order,
                    "odd_hermite_scan": odd_points,
                    "odd_hermite_best": odd_best,
                    "multiscale_scan": multiscale_points,
                    "multiscale_best": multiscale_best,
                    "absolute_error_reduction_fraction": (
                        1
                        - abs(
                            multiscale_best[
                                "error_vs_independent_numerical_reference"
                            ]
                        )
                        / abs(
                            odd_best[
                                "error_vs_independent_numerical_reference"
                            ]
                        )
                    ),
                }
            )
        results[f"n{particles}"] = {
            "reference_energy": (
                N2_CONTINUUM_REFERENCE
                if particles == 2
                else N4_EXTERIOR_D14_REFERENCE
            ),
            "reference_is_numerical_not_continuum_bound": particles == 4,
            "matched_order_results": order_records,
        }
    return results


def _structured_small_global_audits() -> list[dict[str, object]]:
    audits = []
    for particles in [3, 4, 5]:
        raw = ordered_continuous_fourier_hamiltonian_mpo(
            particles,
            3,
            0.8,
            24,
            distance_basis="multiscale_odd_hermite",
            distance_scale_ratio=2.5,
            local_quadrature_order=96,
        )
        reference, ranks, discarded = compress_mpo(raw, 32)
        structured, diagnostics = (
            ordered_continuous_fourier_hamiltonian_compressed_mpo(
                particles,
                3,
                0.8,
                24,
                32,
                distance_basis="multiscale_odd_hermite",
                distance_scale_ratio=2.5,
                local_quadrature_order=96,
            )
        )
        reference_dense = reference.to_dense()
        difference = structured.to_dense() - reference_dense
        audits.append(
            {
                "particles": particles,
                "basis_order": 3,
                "fourier_order": 24,
                "maximum_bond": 32,
                "raw_then_compress_ranks": list(ranks),
                "structured_ranks": list(diagnostics["retained_ranks"]),
                "raw_then_compress_local_discarded_norm": float(discarded),
                "structured_local_discarded_norm": float(
                    diagnostics[
                        "local_discarded_norm_not_global_certificate"
                    ]
                ),
                "global_relative_frobenius_error": float(
                    torch.linalg.matrix_norm(difference)
                    / torch.linalg.matrix_norm(reference_dense)
                ),
                "global_maximum_absolute_error": float(
                    torch.max(torch.abs(difference))
                ),
            }
        )
    return audits


def _resource_point(
    particles: int,
    basis_order: int,
    maximum_bond: int,
) -> dict[str, object]:
    mpo, diagnostics = ordered_continuous_fourier_hamiltonian_compressed_mpo(
        particles,
        basis_order,
        0.5,
        96,
        maximum_bond,
        distance_basis="multiscale_odd_hermite",
        distance_scale_ratio=2.5,
        local_quadrature_order=192 if basis_order == 10 else 160,
    )
    theoretical = diagnostics["theoretical_raw_tensor_elements"]
    intermediate = diagnostics["maximum_intermediate_tensor_elements"]
    return {
        "particles": particles,
        "basis_order": basis_order,
        "maximum_bond": maximum_bond,
        "dense_raw_fourier_bulk_materialized": diagnostics[
            "dense_raw_fourier_bulk_materialized"
        ],
        "theoretical_raw_maximum_bond": diagnostics[
            "theoretical_raw_maximum_bond"
        ],
        "theoretical_raw_tensor_elements": theoretical,
        "maximum_build_intermediate_tensor_elements": intermediate,
        "build_intermediate_reduction_fraction_vs_raw": 1
        - intermediate / theoretical,
        "stored_compressed_tensor_elements": sum(
            tensor.numel() for tensor in mpo.tensors
        ),
        "retained_ranks": list(diagnostics["retained_ranks"]),
    }


def _n6_d10_bond_convergence() -> dict[str, object]:
    generator = torch.Generator().manual_seed(1818)
    vector = torch.randn(10**6, generator=generator, dtype=torch.float64)
    actions = {}
    point_records = []
    for maximum_bond in [128, 192]:
        mpo, diagnostics = (
            ordered_continuous_fourier_hamiltonian_compressed_mpo(
                6,
                10,
                0.5,
                96,
                maximum_bond,
                distance_basis="multiscale_odd_hermite",
                distance_scale_ratio=2.5,
                local_quadrature_order=192,
            )
        )
        actions[maximum_bond] = mpo_product_basis_matvec(mpo, vector)
        point_records.append(
            {
                "maximum_bond": maximum_bond,
                "retained_ranks": list(diagnostics["retained_ranks"]),
                "local_discarded_norm_not_global_certificate": float(
                    diagnostics[
                        "local_discarded_norm_not_global_certificate"
                    ]
                ),
                "maximum_build_intermediate_tensor_elements": diagnostics[
                    "maximum_intermediate_tensor_elements"
                ],
            }
        )
    difference = actions[128] - actions[192]
    return {
        "seed": 1818,
        "product_basis_dimension": 10**6,
        "dense_hamiltonian_materialized": False,
        "points": point_records,
        "bond_128_vs_192_global_action_relative_difference": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(actions[192])
        ),
        "bond_128_vs_192_global_action_maximum_absolute_difference": float(
            torch.max(torch.abs(difference))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase18_basis_structured_controls.json"
        ),
    )
    parser.add_argument(
        "--skip-large-action",
        action="store_true",
        help="development-only shortcut; formal records must not use it",
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    record: dict[str, object] = {
        "schema_version": 1,
        "experiment": "phase18_multiscale_basis_and_structured_mpo_controls",
        "dtype": "float64",
        "basis_analytic_controls": _basis_analytic_controls(),
        "matched_n2_n4_basis_controls": _matched_basis_controls(),
        "structured_small_system_global_audits": (
            _structured_small_global_audits()
        ),
        "structured_resource_points": [
            _resource_point(6, 8, 96),
            _resource_point(8, 10, 128),
        ],
    }
    if not arguments.skip_large_action:
        record["n6_d10_mpo_bond_convergence"] = (
            _n6_d10_bond_convergence()
        )
    record["elapsed_seconds"] = time.perf_counter() - started
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    small_audits_pass = all(
        point["global_relative_frobenius_error"] < 1e-11
        for point in record["structured_small_system_global_audits"]
    )
    basis_controls_pass = max(
        record["basis_analytic_controls"][
            "analytic_vs_independent_quadrature_maximum_absolute_residuals"
        ].values()
    ) < 1e-6
    return 0 if small_audits_pass and basis_controls_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
