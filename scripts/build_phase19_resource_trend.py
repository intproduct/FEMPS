"""Assemble the Phase 19 N=2,4,6,8 accuracy/resource reassessment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stored_mpo_elements(
    particles: int, basis_order: int, ranks: list[int]
) -> int:
    if len(ranks) != particles - 1:
        raise ValueError("an N-site MPO must record N-1 internal ranks")
    extended = [1, *ranks, 1]
    local_operator_elements = basis_order**2
    return sum(
        extended[index]
        * extended[index + 1]
        * local_operator_elements
        for index in range(particles)
    )


def _rebuild_small_system_ranks(
    particles: int, point: dict[str, object]
) -> list[int]:
    """Rebuild a bounded N=2/4 MPO so its full rank chain is explicit."""

    from femps.baselines.ordered_continuous_fourier import (
        ordered_continuous_fourier_hamiltonian_compressed_mpo,
    )

    basis_order = point["basis_order"]
    _, diagnostics = ordered_continuous_fourier_hamiltonian_compressed_mpo(
        particles,
        basis_order,
        point["scale"],
        96,
        max(64, basis_order**2),
        distance_basis=point["basis"],
        distance_scale_ratio=point["scale_ratio"],
        local_quadrature_order=160,
    )
    ranks = list(diagnostics["retained_ranks"])
    if max(ranks) != point["compressed_mpo_maximum_bond"]:
        raise RuntimeError("rebuilt small-system MPO rank disagrees with source")
    return ranks


def _point(
    *,
    particles: int,
    basis_order: int,
    energy: float,
    reference_energy: float,
    reference_description: str,
    reference_is_numerical_not_continuum_bound: bool,
    ranks: list[int],
    method: str,
    elapsed_seconds: float | None,
    peak_cuda_memory_bytes: int | None,
    source_record: str,
) -> dict[str, object]:
    stored = _stored_mpo_elements(particles, basis_order, ranks)
    error = abs(energy - reference_energy)
    return {
        "particles": particles,
        "basis_order": basis_order,
        "energy": energy,
        "reference_energy": reference_energy,
        "absolute_reference_error": error,
        "reference_description": reference_description,
        "reference_is_numerical_not_continuum_bound": (
            reference_is_numerical_not_continuum_bound
        ),
        "method": method,
        "mpo_internal_ranks": ranks,
        "stored_mpo_tensor_elements": stored,
        "absolute_reference_error_per_stored_mpo_element_diagnostic": (
            error / stored
        ),
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "source_record": source_record,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase18-controls",
        type=Path,
        default=Path(
            "docs/experiments/results/phase18_basis_structured_controls.json"
        ),
    )
    parser.add_argument(
        "--phase18-n6-n8",
        type=Path,
        default=Path(
            "docs/experiments/results/phase18_multiscale_n6_n8.json"
        ),
    )
    parser.add_argument(
        "--phase19-n8",
        type=Path,
        default=Path(
            "docs/experiments/results/phase19_n8_d12_multiscale.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase19_accuracy_resource_trend.json"
        ),
    )
    arguments = parser.parse_args()

    controls = _load(arguments.phase18_controls)
    phase18 = _load(arguments.phase18_n6_n8)
    phase19 = _load(arguments.phase19_n8)

    n2_group = controls["matched_n2_n4_basis_controls"]["n2"]
    n2_order = max(
        n2_group["matched_order_results"],
        key=lambda point: point["basis_order"],
    )
    n2 = n2_order["multiscale_best"]

    n4_group = controls["matched_n2_n4_basis_controls"]["n4"]
    n4_order = max(
        n4_group["matched_order_results"],
        key=lambda point: point["basis_order"],
    )
    n4 = n4_order["multiscale_best"]

    n6_runs = phase18["n6"]["blind_multiseed_D10"]
    n6 = min(n6_runs, key=lambda point: point["final_energy"])
    n6_reference = phase18["n6"]["exterior_D12_numerical_reference"]
    n8 = phase19["production"]
    n8_reference = phase19["exterior_D14_Q160_numerical_reference"]
    n2_ranks = _rebuild_small_system_ranks(2, n2)
    n4_ranks = _rebuild_small_system_ranks(4, n4)

    points = [
        _point(
            particles=2,
            basis_order=n2["basis_order"],
            energy=n2["ground_energy"],
            reference_energy=n2_group["reference_energy"],
            reference_description=(
                "independent relative-coordinate continuum numerical value"
            ),
            reference_is_numerical_not_continuum_bound=False,
            ranks=n2_ranks,
            method=n2["truth_method"],
            elapsed_seconds=None,
            peak_cuda_memory_bytes=None,
            source_record=str(arguments.phase18_controls),
        ),
        _point(
            particles=4,
            basis_order=n4["basis_order"],
            energy=n4["ground_energy"],
            reference_energy=n4_group["reference_energy"],
            reference_description="exterior D=14 numerical reference",
            reference_is_numerical_not_continuum_bound=True,
            ranks=n4_ranks,
            method=n4["truth_method"],
            elapsed_seconds=n4["elapsed_seconds"],
            peak_cuda_memory_bytes=None,
            source_record=str(arguments.phase18_controls),
        ),
        _point(
            particles=6,
            basis_order=n6["basis_order"],
            energy=n6["final_energy"],
            reference_energy=n6_reference,
            reference_description="exterior D=12 numerical reference",
            reference_is_numerical_not_continuum_bound=True,
            ranks=list(n6["mpo_compression_ranks"]),
            method="native global-AD MPS/MPO production",
            elapsed_seconds=n6["elapsed_seconds"],
            peak_cuda_memory_bytes=n6["peak_cuda_memory_bytes"],
            source_record=str(arguments.phase18_n6_n8),
        ),
        _point(
            particles=8,
            basis_order=12,
            energy=n8["final_energy"],
            reference_energy=n8_reference,
            reference_description="exterior D=14 numerical reference",
            reference_is_numerical_not_continuum_bound=True,
            ranks=list(n8["mpo_compression_ranks"]),
            method="native global-AD MPS/MPO production",
            elapsed_seconds=n8["elapsed_seconds"],
            peak_cuda_memory_bytes=n8["peak_cuda_memory_bytes"],
            source_record=str(arguments.phase19_n8),
        ),
    ]
    by_particles = {point["particles"]: point for point in points}
    n6_point = by_particles[6]
    n8_point = by_particles[8]
    record = {
        "schema_version": 1,
        "experiment": "phase19_n2_n4_n6_n8_accuracy_resource_reassessment",
        "scope": (
            "representative best controlled points; descriptive comparison, "
            "not a matched-order scaling fit"
        ),
        "source_records": {
            "phase18_controls": str(arguments.phase18_controls),
            "phase18_n6_n8": str(arguments.phase18_n6_n8),
            "phase19_n8_D12": str(arguments.phase19_n8),
        },
        "points": points,
        "n8_vs_n6_production_ratios": {
            "stored_mpo_tensor_elements": (
                n8_point["stored_mpo_tensor_elements"]
                / n6_point["stored_mpo_tensor_elements"]
            ),
            "elapsed_seconds": (
                n8_point["elapsed_seconds"] / n6_point["elapsed_seconds"]
            ),
            "peak_cuda_memory_bytes": (
                n8_point["peak_cuda_memory_bytes"]
                / n6_point["peak_cuda_memory_bytes"]
            ),
            "absolute_reference_error": (
                n8_point["absolute_reference_error"]
                / n6_point["absolute_reference_error"]
            ),
        },
        "interpretation": {
            "accuracy_per_entry_is_descriptive_only": True,
            "heterogeneous_solver_resources_are_not_directly_comparable": True,
            "reference_errors_do_not_define_continuum_error_bounds": True,
            "favorable_asymptotic_scaling_inferred": False,
            "n10_admitted": False,
            "n10_decision": (
                "defer: the refined N=8 point is controlled, but reference "
                "error still grows with particle number at representative "
                "local orders and no continuum/asymptotic resource law exists"
            ),
        },
    }
    if not all(math.isfinite(point["energy"]) for point in points):
        raise RuntimeError("resource trend contains a non-finite energy")
    record["resource_reassessment_complete"] = True
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
