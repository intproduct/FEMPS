"""Independently reconstruct the Phase 32 N=6 convergence tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from femps.exterior import (
    antisymmetry_residual,
    diagonal_path_structural_counts,
    exterior_coefficients_to_tensor,
    particle_tt_ranks_exterior_coefficients,
)
from femps.hamiltonians import (
    antisymmetric_many_body_hamiltonian,
    antisymmetric_many_body_hamiltonian_dense_two_body,
    harmonic_pair_hamiltonian,
    soft_coulomb_dense_quadrature,
    soft_coulomb_operator,
)


PARTICLES = 6
DIMENSIONS = (8, 10, 12)
TERMS = (1, 2, 4)
SEEDS = {
    "D10_K1": 3201,
    "D10_K2": 3202,
    "D10_K4": 3204,
    "D8_K4": 3284,
    "D12_K4": 3212,
}
POINT_IDENTITIES = {
    "N6_D10_K1_seed3201_blind": (10, 1),
    "N6_D10_K2_seed3202_from_K1": (10, 2),
    "N6_D10_K4_seed3204_from_K2": (10, 4),
    "N6_D8_K4_seed3284_blind": (8, 4),
    "N6_D12_K4_seed3212_from_D10": (12, 4),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coefficients(raw: list[list[float]]) -> torch.Tensor:
    return torch.tensor(
        [complex(real, imaginary) for real, imaginary in raw],
        dtype=torch.complex128,
    )


def _close(observed: float, expected: float, label: str, atol: float = 2e-10) -> None:
    if not math.isclose(observed, expected, rel_tol=2e-12, abs_tol=atol):
        raise AssertionError(f"{label} is inconsistent: {observed} != {expected}")


def _tt_storage(dimension: int, ranks: tuple[int, ...]) -> int:
    extended = (1,) + ranks + (1,)
    return sum(
        extended[index] * dimension * extended[index + 1]
        for index in range(len(extended) - 1)
    )


def _state_metrics(
    coefficients: torch.Tensor,
    hamiltonian: torch.Tensor,
    dimension: int,
) -> dict:
    norm = torch.vdot(coefficients, coefficients).real
    acted = hamiltonian @ coefficients
    energy = (torch.vdot(coefficients, acted) / norm).real
    residual = acted - energy * coefficients
    variance = torch.vdot(residual, residual).real / norm
    ranks = particle_tt_ranks_exterior_coefficients(
        coefficients, dimension, PARTICLES
    )
    return {
        "energy": float(energy),
        "variance": float(variance),
        "norm_error": float(abs(norm - 1.0)),
        "ranks": ranks,
        "storage": _tt_storage(dimension, ranks),
    }


def verify_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["schema_version"] != 1:
        raise AssertionError("unexpected schema")
    if artifact["experiment"] != "phase32_N6_independent_D_K_convergence":
        raise AssertionError("unexpected experiment")
    if artifact["evidence_level"] != "numerical":
        raise AssertionError("floating-point results must remain numerical evidence")
    if "not an asymptotic scaling fit" not in artifact["scientific_boundary"]:
        raise AssertionError("scientific boundary was weakened")
    if artifact["model"] != {
        "N": PARTICLES,
        "dimensions": list(DIMENSIONS),
        "terms": list(TERMS),
        "Q": 128,
        "Q_check": 160,
        "coupling": 1.0,
        "softening": 1.0,
    }:
        raise AssertionError("registered physical model changed")
    if artifact["registered_config"] != {
        "seeds": SEEDS,
        "steps": 160,
        "lbfgs_steps": 80,
        "learning_rate": 1e-3,
        "final_learning_rate": 1e-5,
        "device": "cpu",
        "truth_state_initialization": False,
    }:
        raise AssertionError("registered seeds or optimization budget changed")
    if artifact["thresholds"] != {
        "energy_monotonicity_tolerance": 1e-9,
        "norm_error": 1e-10,
        "antisymmetry_residual": 1e-12,
        "operator_factorization_error": 1e-11,
        "quadrature_relative_change": 2e-12,
        "peak_cpu_rss_bytes": 2 * 1024**3,
        "peak_cuda_memory_bytes": 4 * 1024**3,
        "wall_time_seconds_per_point": 600.0,
    }:
        raise AssertionError("registered thresholds changed")

    for source in artifact["source_records"].values():
        source_path = Path(source["path"])
        if _sha256(source_path) != source["sha256"]:
            raise AssertionError("a registered source artifact changed")
    stability = artifact["source_records"]["decisive_K4_multiseed"]
    if stability["seeds"] != [31, 37, 43] or not stability["pass"]:
        raise AssertionError("decisive K=4 blind multiseed gate changed")

    basis_by_d = {point["D"]: point for point in artifact["basis_audits"]}
    if tuple(basis_by_d) != DIMENSIONS:
        raise AssertionError("basis axis changed")
    dense_hamiltonians = {}
    factorized_hamiltonians = {}
    for dimension in DIMENSIONS:
        audit = basis_by_d[dimension]
        one_body = harmonic_pair_hamiltonian(
            dimension, kappa=0.0, dtype=torch.complex128, device="cpu"
        )[0]
        dense_pair = soft_coulomb_dense_quadrature(
            dimension,
            quadrature_order=128,
            coupling=1.0,
            softening=1.0,
            dtype=torch.complex128,
            device="cpu",
        )
        dense_pair_check = soft_coulomb_dense_quadrature(
            dimension,
            quadrature_order=160,
            coupling=1.0,
            softening=1.0,
            dtype=torch.complex128,
            device="cpu",
        )
        interaction, diagnostics = soft_coulomb_operator(
            dimension,
            quadrature_order=128,
            coupling=1.0,
            softening=1.0,
            relative_threshold=1e-13,
            factorization_backend="physical",
            dtype=torch.complex128,
            device="cpu",
        )
        quadrature_change = float(
            torch.linalg.vector_norm(dense_pair - dense_pair_check)
            / torch.linalg.vector_norm(dense_pair_check)
        )
        _close(
            audit["quadrature_relative_change_Q128_vs_Q160"],
            quadrature_change,
            f"D={dimension} quadrature change",
            atol=1e-16,
        )
        if audit["physical_operator_svd_rank"] != diagnostics.retained_rank:
            raise AssertionError("physical operator-SVD rank changed")
        _close(
            audit["physical_operator_svd_relative_error"],
            diagnostics.dense_relative_factorization_error,
            f"D={dimension} factorization error",
            atol=1e-16,
        )
        if audit["exterior_ci_dimension"] != math.comb(dimension, PARTICLES):
            raise AssertionError("CI dimension changed")
        if audit["forbidden_particle_tensor_coefficients"] != dimension**PARTICLES:
            raise AssertionError("D^N count changed")

        dense_hamiltonian = antisymmetric_many_body_hamiltonian_dense_two_body(
            one_body, PARTICLES, dense_pair
        )
        factorized_hamiltonian = antisymmetric_many_body_hamiltonian(
            one_body, PARTICLES, interaction
        )
        dense_hamiltonians[dimension] = dense_hamiltonian
        factorized_hamiltonians[dimension] = factorized_hamiltonian
        ci_coefficients = _coefficients(
            audit["direct_ci"]["raw_exterior_coefficients"]
        )
        ci_metrics = _state_metrics(ci_coefficients, dense_hamiltonian, dimension)
        _close(audit["direct_ci"]["energy"], ci_metrics["energy"], "CI energy")
        _close(
            audit["direct_ci"]["energy_variance"],
            ci_metrics["variance"],
            "CI variance",
        )
        _close(
            audit["direct_ci"]["norm_error"], ci_metrics["norm_error"], "CI norm"
        )
        if tuple(audit["direct_ci"]["ordinary_particle_tt_ranks"]) != ci_metrics[
            "ranks"
        ]:
            raise AssertionError("CI ordinary particle-TT ranks changed")
        if (
            audit["direct_ci"]["ordinary_particle_tt_storage_scalars"]
            != ci_metrics["storage"]
        ):
            raise AssertionError("CI TT storage changed")

        slater = torch.zeros(math.comb(dimension, PARTICLES), dtype=torch.complex128)
        slater[0] = 1.0
        slater_metrics = _state_metrics(slater, dense_hamiltonian, dimension)
        _close(
            audit["reference_slater"]["energy"],
            slater_metrics["energy"],
            "Slater energy",
        )
        _close(
            audit["reference_slater"]["energy_variance"],
            slater_metrics["variance"],
            "Slater variance",
        )
        if tuple(
            audit["reference_slater"]["ordinary_particle_tt_ranks"]
        ) != slater_metrics["ranks"]:
            raise AssertionError("Slater ordinary particle-TT ranks changed")

    point_by_id = {point["point_id"]: point for point in artifact["points"]}
    if set(point_by_id) != set(POINT_IDENTITIES):
        raise AssertionError("registered point identities changed")
    recomputed_rows = {}
    for point_id, (dimension, terms) in POINT_IDENTITIES.items():
        point = point_by_id[point_id]
        config = point["config"]
        if (
            config["basis_order"] != dimension
            or config["particles"] != PARTICLES
            or config["terms"] != terms
            or config["steps"] != 160
            or config["lbfgs_refinement_steps"] != 80
            or config["device"] != "cpu"
        ):
            raise AssertionError("point configuration changed")
        if point["evidence_level"] != "numerical" or not point["completed"]:
            raise AssertionError("point evidence/completion changed")
        if point["initialization_lineage"]["truth_state_used"]:
            raise AssertionError("truth-state initialization is forbidden")
        expected_counts = diagonal_path_structural_counts(
            PARTICLES,
            dimension,
            terms,
            basis_by_d[dimension]["physical_operator_svd_rank"],
        )
        if point["structural_counts"] != expected_counts:
            raise AssertionError("structural counts changed")
        coefficients = _coefficients(point["raw_exterior_coefficients"])
        metrics = _state_metrics(
            coefficients, dense_hamiltonians[dimension], dimension
        )
        factor_metrics = _state_metrics(
            coefficients, factorized_hamiltonians[dimension], dimension
        )
        _close(point["dense_quadrature_energy"], metrics["energy"], "point energy")
        _close(
            point["dense_quadrature_energy_variance"],
            metrics["variance"],
            "point variance",
        )
        _close(
            point["dense_quadrature_norm_error"], metrics["norm_error"], "point norm"
        )
        _close(point["energy"], factor_metrics["energy"], "factorized point energy")
        _close(
            point["factorized_vs_dense_energy_difference"],
            factor_metrics["energy"] - metrics["energy"],
            "factorized/dense difference",
        )
        error = metrics["energy"] - basis_by_d[dimension]["direct_ci"]["energy"]
        _close(point["error_vs_dense_quadrature_ci"], error, "point CI error")
        if tuple(point["ordinary_particle_tt_ranks_compact"]) != metrics["ranks"]:
            raise AssertionError("point ordinary particle-TT ranks changed")
        if point["ordinary_particle_tt_storage_scalars"] != metrics["storage"]:
            raise AssertionError("point TT storage changed")
        if point["femps_stored_parameter_scalars"] != terms * dimension * 6 + terms:
            raise AssertionError("FEMPS storage changed")
        admitted = point["validation_materialized_antisymmetry_residual"] is not None
        if admitted:
            tensor = exterior_coefficients_to_tensor(
                coefficients, dimension, PARTICLES
            )
            residual = float(antisymmetry_residual(tensor))
            _close(
                point["validation_materialized_antisymmetry_residual"],
                residual,
                "materialized antisymmetry residual",
                atol=1e-15,
            )
        recomputed_rows[point_id] = metrics

    def check_axis(axis: list[dict], expected_ids: list[str]) -> None:
        if [row["point_id"] for row in axis] != expected_ids:
            raise AssertionError("axis point order changed")
        for row in axis:
            point = point_by_id[row["point_id"]]
            metrics = recomputed_rows[row["point_id"]]
            _close(row["energy"], metrics["energy"], "axis energy")
            _close(row["energy_variance"], metrics["variance"], "axis variance")
            if row["elapsed_seconds"] != point["total_elapsed_seconds_this_call"]:
                raise AssertionError("axis timing was not copied from the raw point")
            if row["peak_cpu_rss_bytes"] != point["peak_cpu_rss_bytes"]:
                raise AssertionError("axis RSS was not copied from the raw point")

    check_axis(
        artifact["correlation_axis"],
        [
            "N6_D10_K1_seed3201_blind",
            "N6_D10_K2_seed3202_from_K1",
            "N6_D10_K4_seed3204_from_K2",
        ],
    )
    check_axis(
        artifact["basis_axis"],
        [
            "N6_D8_K4_seed3284_blind",
            "N6_D10_K4_seed3204_from_K2",
            "N6_D12_K4_seed3212_from_D10",
        ],
    )

    tolerance = artifact["thresholds"]["energy_monotonicity_tolerance"]
    k_energies = [row["energy"] for row in artifact["correlation_axis"]]
    d_energies = [row["energy"] for row in artifact["basis_axis"]]
    k_pass = all(b <= a + tolerance for a, b in zip(k_energies, k_energies[1:]))
    d_pass = all(b <= a + tolerance for a, b in zip(d_energies, d_energies[1:]))
    structural_pass = all(
        point["structural_antisymmetry_residual"] <= 1e-12
        and point["structural_counts"]["enumerated_virtual_paths"] == 0
        and point["structural_counts"]["materialized_particle_coefficients"] == 0
        and point["dense_quadrature_norm_error"] <= 1e-10
        for point in artifact["points"]
    )
    resource_pass = all(
        point["total_elapsed_seconds_this_call"] <= 600.0
        and point["peak_cpu_rss_bytes"] <= 2 * 1024**3
        and point["peak_cuda_memory_bytes"] is None
        for point in artifact["points"]
    )
    recomputed_acceptance = {
        "structural_pass": structural_pass,
        "materialization_validation_pass": all(
            point_by_id[point_id]["validation_materialized_antisymmetry_residual"]
            <= 1e-12
            for point_id in (
                "N6_D8_K4_seed3284_blind",
                "N6_D10_K4_seed3204_from_K2",
            )
        ),
        "K_axis_pass": k_pass,
        "D_axis_pass": d_pass,
        "operator_pass": all(
            basis_by_d[d]["physical_operator_svd_relative_error"] <= 1e-11
            and basis_by_d[d]["quadrature_relative_change_Q128_vs_Q160"] <= 2e-12
            for d in DIMENSIONS
        ),
        "resource_pass": resource_pass,
    }
    recomputed_acceptance["phase32_convergence_pass"] = all(
        recomputed_acceptance.values()
    )
    if artifact["acceptance"] != recomputed_acceptance:
        raise AssertionError("acceptance record is inconsistent")
    if not recomputed_acceptance["phase32_convergence_pass"]:
        raise AssertionError("Phase 32 convergence gate did not pass")
    return {
        "verified": True,
        "K_energies": k_energies,
        "D_energies": d_energies,
        "D12_error_vs_direct_ci": point_by_id[
            "N6_D12_K4_seed3212_from_D10"
        ]["error_vs_dense_quadrature_ci"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        type=Path,
        nargs="?",
        default=Path("docs/experiments/results/phase32_n6_convergence.json"),
    )
    print(json.dumps(verify_artifact(parser.parse_args().artifact), indent=2))


if __name__ == "__main__":
    main()
