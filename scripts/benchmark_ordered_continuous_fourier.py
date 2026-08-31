"""Phase 17 controls for the unbounded Fourier--Bessel interaction MPO."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from femps.basis.odd_hermite import (
    odd_hermite_basis_values,
)
from femps.baselines.ordered_continuous_fourier import (
    ordered_continuous_fourier_hamiltonian_mpo,
    ordered_continuous_fourier_soft_coulomb_mpo,
    ordered_continuous_fourier_soft_coulomb_pair_mpo,
    soft_coulomb_fourier_sampled_error,
)
from femps.baselines.ordered_continuous_mpo import (
    ordered_continuous_soft_coulomb_hamiltonian_mpo,
)
from femps.baselines.ordered_distance_mpo import compress_mpo
from femps.benchmarks.mpo_truth import (
    lowest_mpo_eigenpair,
    mpo_product_basis_matvec,
)


N2_CONTINUUM_REFERENCE = 2.553831733978763
N4_EXTERIOR_D14_REFERENCE = 11.023082853674637


def _maximum_bond(mpo) -> int:
    return max(max(tensor.shape[:2]) for tensor in mpo.tensors)


def _half_line_quadrature(order: int, scale: float, count: int):
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(count)
    cutoff = scale * (math.sqrt(4 * order + 2) + 8)
    nodes = torch.from_numpy((raw_nodes + 1) * cutoff / 2)
    weights = torch.from_numpy(raw_weights * cutoff / 2)
    basis = odd_hermite_basis_values(order, nodes, scale)
    return nodes, weights, basis


def _separation_controls() -> dict[str, object]:
    scalar = [
        {
            "fourier_order": order,
            "maximum_sampled_error_on_0_to_12": (
                soft_coulomb_fourier_sampled_error(12.0, order)
            ),
        }
        for order in [64, 96, 128, 160, 192]
    ]

    one_nodes, one_weights, one_basis = _half_line_quadrature(5, 0.8, 500)
    one_expected = torch.einsum(
        "xm,x,x,xn->mn",
        one_basis,
        one_weights,
        torch.rsqrt(one_nodes.square() + 1),
        one_basis,
    )
    one_gap = []
    for order in [64, 96, 128, 160]:
        mpo = ordered_continuous_fourier_soft_coulomb_pair_mpo(
            2, 5, 0.8, 0, 1, order, local_quadrature_order=192
        )
        observed = mpo.to_dense().reshape(5, 5, 5, 5)[0, :, 0, :]
        one_gap.append(
            {
                "fourier_order": order,
                "frobenius_error_vs_direct_half_line_quadrature": float(
                    torch.linalg.matrix_norm(observed - one_expected)
                ),
            }
        )

    two_nodes, two_weights, two_basis = _half_line_quadrature(3, 0.8, 180)
    densities = torch.einsum("xm,xn->xmn", two_basis, two_basis)
    potential = torch.rsqrt(
        (two_nodes[:, None] + two_nodes[None, :]).square() + 1
    )
    relative = torch.einsum(
        "xmn,yab,xy,x,y->manb",
        densities,
        densities,
        potential,
        two_weights,
        two_weights,
    ).reshape(9, 9)
    two_expected = torch.kron(torch.eye(3), relative)
    two_gap = []
    for order in [64, 96, 128, 160]:
        mpo = ordered_continuous_fourier_soft_coulomb_pair_mpo(
            3, 3, 0.8, 0, 2, order, local_quadrature_order=192
        )
        two_gap.append(
            {
                "fourier_order": order,
                "frobenius_error_vs_direct_two_gap_quadrature": float(
                    torch.linalg.matrix_norm(mpo.to_dense() - two_expected)
                ),
            }
        )
    return {
        "transform": (
            "1/sqrt(s^2+a^2)=(2/pi) integral_0^inf K0(a*k) cos(k*s) dk"
        ),
        "dimensionless_frequency_cutoff": 30.0,
        "scalar_sampled": scalar,
        "one_gap_projected_operator": one_gap,
        "two_gap_projected_operator": two_gap,
    }


def _n2_fourier_energy(
    basis_order: int,
    scale: float,
    fourier_order: int = 160,
    quadrature_order: int = 192,
) -> float:
    mpo = ordered_continuous_fourier_hamiltonian_mpo(
        2,
        basis_order,
        scale,
        fourier_order,
        local_quadrature_order=quadrature_order,
    )
    return float(torch.linalg.eigvalsh(mpo.to_dense())[0])


def _n2_controls() -> dict[str, object]:
    basis_comparison = []
    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    for basis_order in [4, 6, 8, 10, 12]:
        odd_points = [
            {
                "scale": scale,
                "ground_energy": _n2_fourier_energy(basis_order, scale),
            }
            for scale in scales
        ]
        odd_best = min(odd_points, key=lambda point: point["ground_energy"])
        sine = ordered_continuous_soft_coulomb_hamiltonian_mpo(
            2,
            basis_order,
            9.0,
            32,
            interaction_quadrature_order=220,
        )
        sine_energy = float(torch.linalg.eigvalsh(sine.to_dense())[0])
        basis_comparison.append(
            {
                "basis_order": basis_order,
                "odd_hermite_scale_scan": odd_points,
                "odd_hermite_best": {
                    **odd_best,
                    "error_vs_independent_continuum_reference": (
                        odd_best["ground_energy"] - N2_CONTINUUM_REFERENCE
                    ),
                },
                "dirichlet_sine_box_9": {
                    "ground_energy": sine_energy,
                    "error_vs_independent_continuum_reference": (
                        sine_energy - N2_CONTINUUM_REFERENCE
                    ),
                },
            }
        )

    order_scan = []
    for order in [64, 80, 96, 112, 128, 160, 192, 256]:
        order_scan.append(
            {
                "fourier_order": order,
                "ground_energy": _n2_fourier_energy(12, 1.0, order),
            }
        )
    reference = order_scan[-1]["ground_energy"]
    for point in order_scan:
        point["difference_vs_order_256"] = point["ground_energy"] - reference

    quadrature_scan = []
    for quadrature_order in [96, 128, 160, 192, 256]:
        quadrature_scan.append(
            {
                "local_quadrature_order": quadrature_order,
                "ground_energy": _n2_fourier_energy(
                    12, 1.0, 160, quadrature_order
                ),
            }
        )
    quadrature_reference = quadrature_scan[-1]["ground_energy"]
    for point in quadrature_scan:
        point["difference_vs_q256"] = (
            point["ground_energy"] - quadrature_reference
        )
    return {
        "independent_continuum_reference": N2_CONTINUUM_REFERENCE,
        "reference_provenance": (
            "second-order Richardson extrapolation of an independent "
            "relative-coordinate half-line finite-difference solve"
        ),
        "matched_basis_order_comparison": basis_comparison,
        "fourier_order_at_D12_scale_1": order_scan,
        "local_quadrature_at_D12_scale_1_M160": quadrature_scan,
    }


def _n4_fourier_point(
    basis_order: int,
    scale: float,
    fourier_order: int = 96,
    quadrature_order: int = 160,
    seed: int = 1730,
) -> dict[str, object]:
    raw = ordered_continuous_fourier_hamiltonian_mpo(
        4,
        basis_order,
        scale,
        fourier_order,
        local_quadrature_order=quadrature_order,
    )
    compression_bond = max(64, basis_order**2)
    compressed, ranks, discarded = compress_mpo(raw, compression_bond)
    energy, _, lanczos = lowest_mpo_eigenpair(compressed, seed=seed)
    return {
        "basis_order": basis_order,
        "scale": scale,
        "fourier_order": fourier_order,
        "local_quadrature_order": quadrature_order,
        "raw_mpo_max_bond": _maximum_bond(raw),
        "compressed_mpo_ranks": list(ranks),
        "compression_local_discarded_norm_not_global_certificate": float(
            discarded
        ),
        "ground_energy": energy,
        "error_vs_exterior_D14_reference": energy - N4_EXTERIOR_D14_REFERENCE,
        **lanczos,
    }


def _n4_sine_point(basis_order: int, seed: int) -> dict[str, object]:
    mpo = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        4,
        basis_order,
        4.5,
        20,
        interaction_quadrature_order=160,
    )
    energy, _, lanczos = lowest_mpo_eigenpair(mpo, seed=seed)
    return {
        "basis_order": basis_order,
        "distance_box": 4.5,
        "ground_energy": energy,
        "error_vs_exterior_D14_reference": energy - N4_EXTERIOR_D14_REFERENCE,
        **lanczos,
    }


def _n4_controls() -> dict[str, object]:
    basis_comparison = []
    for basis_order in [4, 6, 8]:
        odd_points = [
            _n4_fourier_point(
                basis_order,
                scale,
                seed=1740 + 10 * basis_order + scale_index,
            )
            for scale_index, scale in enumerate([0.6, 0.7, 0.8, 0.9, 1.0])
        ]
        odd_best = min(odd_points, key=lambda point: point["ground_energy"])
        basis_comparison.append(
            {
                "basis_order": basis_order,
                "odd_hermite_scale_scan": odd_points,
                "odd_hermite_best": odd_best,
                "dirichlet_sine": _n4_sine_point(
                    basis_order, seed=1800 + basis_order
                ),
            }
        )

    order_scan = [
        _n4_fourier_point(8, 0.7, order, 160, seed=1810 + order)
        for order in [64, 80, 96, 112, 128]
    ]
    order_reference = order_scan[-1]["ground_energy"]
    for point in order_scan:
        point["difference_vs_order_128"] = (
            point["ground_energy"] - order_reference
        )

    quadrature_scan = [
        _n4_fourier_point(8, 0.7, 96, order, seed=1900 + order)
        for order in [96, 128, 160, 224]
    ]
    quadrature_reference = quadrature_scan[-1]["ground_energy"]
    for point in quadrature_scan:
        point["difference_vs_q224"] = (
            point["ground_energy"] - quadrature_reference
        )
    return {
        "post_run_exterior_D14_reference": N4_EXTERIOR_D14_REFERENCE,
        "reference_is_numerical_not_continuum_bound": True,
        "matched_basis_order_comparison": basis_comparison,
        "fourier_order_at_D8_scale_0_7": order_scan,
        "local_quadrature_at_D8_scale_0_7_M96": quadrature_scan,
    }


def _compression_point(raw, reference, vector, maximum_bond: int):
    compressed, ranks, discarded = compress_mpo(raw, maximum_bond)
    observed = mpo_product_basis_matvec(compressed, vector)
    difference = observed - reference
    return {
        "requested_maximum_bond": maximum_bond,
        "retained_ranks": list(ranks),
        "local_discarded_norm_not_global_certificate": float(discarded),
        "global_random_action_relative_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference)
        ),
        "global_random_action_maximum_absolute_error": float(
            torch.max(torch.abs(difference))
        ),
    }


def _mpo_scaling_and_compression() -> dict[str, object]:
    small_kwargs = dict(
        particles=4,
        basis_order=3,
        distance_scale=0.8,
        fourier_order=24,
        local_quadrature_order=96,
    )
    compact_small = ordered_continuous_fourier_soft_coulomb_mpo(
        **small_kwargs, construction="compact"
    )
    direct_small = ordered_continuous_fourier_soft_coulomb_mpo(
        **small_kwargs, construction="direct_pairs"
    )
    compact_dense = compact_small.to_dense()
    direct_dense = direct_small.to_dense()
    equivalence = {
        "particles": 4,
        "basis_order": 3,
        "fourier_order": 24,
        "global_relative_frobenius_error": float(
            torch.linalg.matrix_norm(compact_dense - direct_dense)
            / torch.linalg.matrix_norm(direct_dense)
        ),
        "maximum_absolute_error": float(
            torch.max(torch.abs(compact_dense - direct_dense))
        ),
    }

    raw_scaling = []
    for particles in range(2, 9):
        compact = ordered_continuous_fourier_soft_coulomb_mpo(
            particles,
            2,
            0.9,
            16,
            local_quadrature_order=64,
            construction="compact",
        )
        direct = ordered_continuous_fourier_soft_coulomb_mpo(
            particles,
            2,
            0.9,
            16,
            local_quadrature_order=64,
            construction="direct_pairs",
        )
        raw_scaling.append(
            {
                "particles": particles,
                "compact_maximum_bond": _maximum_bond(compact),
                "direct_pair_maximum_bond": _maximum_bond(direct),
                "compact_tensor_elements": sum(
                    tensor.numel() for tensor in compact.tensors
                ),
                "direct_pair_tensor_elements": sum(
                    tensor.numel() for tensor in direct.tensors
                ),
            }
        )

    raw_d4 = ordered_continuous_fourier_hamiltonian_mpo(
        4, 4, 0.9, 64, local_quadrature_order=128
    )
    dense_d4 = raw_d4.to_dense()
    dense_norm = torch.linalg.matrix_norm(dense_d4)
    dense_compression = []
    for maximum_bond in [8, 16, 24, 32, 64]:
        compressed, ranks, discarded = compress_mpo(raw_d4, maximum_bond)
        difference = compressed.to_dense() - dense_d4
        dense_compression.append(
            {
                "requested_maximum_bond": maximum_bond,
                "retained_ranks": list(ranks),
                "local_discarded_norm_not_global_certificate": float(discarded),
                "global_relative_frobenius_error": float(
                    torch.linalg.matrix_norm(difference) / dense_norm
                ),
                "global_maximum_absolute_error": float(
                    torch.max(torch.abs(difference))
                ),
            }
        )

    raw_d8 = ordered_continuous_fourier_hamiltonian_mpo(
        4, 8, 0.7, 96, local_quadrature_order=160
    )
    generator = torch.Generator().manual_seed(1701)
    vector = torch.randn(8**4, generator=generator, dtype=torch.float64)
    reference = mpo_product_basis_matvec(raw_d8, vector)
    action_compression = [
        _compression_point(raw_d8, reference, vector, maximum_bond)
        for maximum_bond in [32, 48, 64, 96]
    ]
    return {
        "compact_recurrence": (
            "four real states [1,c,s,T] per Fourier node; bond 4*M is "
            "independent of particle count"
        ),
        "compact_vs_direct_pair_global_audit": equivalence,
        "raw_interaction_scaling_at_D2_M16": raw_scaling,
        "n4_D4_M64_dense_global_compression": {
            "raw_maximum_bond": _maximum_bond(raw_d4),
            "points": dense_compression,
        },
        "n4_D8_M96_random_action_global_compression": {
            "seed": 1701,
            "product_basis_dimension": 8**4,
            "raw_maximum_bond": _maximum_bond(raw_d8),
            "points": action_compression,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase17_unbounded_fourier_controls.json"
        ),
    )
    parser.add_argument(
        "--skip-n4", action="store_true", help="development-only shortcut"
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    record: dict[str, object] = {
        "schema_version": 1,
        "experiment": "phase17_unbounded_fourier_bessel_controls",
        "dtype": "float64",
        "separation_controls": _separation_controls(),
        "n2": _n2_controls(),
        "mpo_scaling_and_compression": _mpo_scaling_and_compression(),
    }
    if not arguments.skip_n4:
        record["n4"] = _n4_controls()
    record["elapsed_seconds"] = time.perf_counter() - started
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
