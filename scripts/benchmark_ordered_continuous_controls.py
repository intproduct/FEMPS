"""Independent error controls for the Phase 16 continuous ordered solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from femps.baselines.ordered_continuous_interaction import (
    soft_coulomb_chebyshev_sampled_error,
)
from femps.baselines.ordered_continuous_mpo import (
    ordered_continuous_noninteracting_mpo,
    ordered_continuous_soft_coulomb_hamiltonian_mpo,
)
from femps.baselines.ordered_functional_mps import (
    particle_tensor_to_mps_tensors,
)
from femps.benchmarks.mpo_truth import mpo_product_basis_matvec
from femps.exterior import antisymmetry_residual
from femps.ordered_sector import extend_from_ordered_sector


N2_CONTINUUM_REFERENCE = 2.553831733978763
N4_EXTERIOR_D14_REFERENCE = 11.023082853674637


def _antisymmetry_audit() -> dict[str, object]:
    generator = torch.Generator().manual_seed(1599)
    ordered_values = torch.randn(15, generator=generator, dtype=torch.float64)
    full_state = extend_from_ordered_sector(ordered_values, 6, 4)
    return {
        "mechanism": "exact signed extension of one ordered chamber",
        "particles": 4,
        "local_grid_dimension": 6,
        "relative_antisymmetry_residual": float(
            antisymmetry_residual(full_state)
        ),
        "collision_amplitude_maximum": float(
            torch.max(
                torch.abs(
                    torch.diagonal(full_state, dim1=0, dim2=1)
                )
            )
        ),
    }


def _small_dense_matvec_audit() -> dict[str, float | int]:
    mpo = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        3, 3, 4.0, 4, interaction_quadrature_order=48
    )
    generator = torch.Generator().manual_seed(1600)
    vector = torch.randn(3**3, generator=generator, dtype=torch.float64)
    observed = mpo_product_basis_matvec(mpo, vector)
    expected = mpo.to_dense() @ vector
    return {
        "particles": 3,
        "basis_order": 3,
        "maximum_absolute_error": float(torch.max(torch.abs(observed - expected))),
    }


def _lowest_mpo_eigenpair(
    mpo,
    *,
    tolerance: float = 2e-10,
    maximum_iterations: int = 1200,
    seed: int = 1610,
) -> tuple[float, torch.Tensor, dict[str, float | int]]:
    """Return the lowest product-basis eigenpair using an independent Lanczos path."""

    from scipy.sparse.linalg import LinearOperator, eigsh

    if any(tensor.device.type != "cpu" for tensor in mpo.tensors):
        raise ValueError("the independent Lanczos audit is CPU-only")
    dimension = mpo.dim**mpo.length
    calls = 0

    def matvec(vector: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        tensor = torch.from_numpy(np.asarray(vector, dtype=np.float64))
        return mpo_product_basis_matvec(mpo, tensor).detach().numpy()

    operator = LinearOperator(
        (dimension, dimension), matvec=matvec, rmatvec=matvec, dtype=np.float64
    )
    initial = np.random.default_rng(seed).normal(size=dimension)
    started = time.perf_counter()
    values, vectors = eigsh(
        operator,
        k=1,
        which="SA",
        v0=initial,
        tol=tolerance,
        maxiter=maximum_iterations,
    )
    elapsed = time.perf_counter() - started
    eigenvector = torch.from_numpy(vectors[:, 0].copy())
    residual = torch.linalg.vector_norm(
        mpo_product_basis_matvec(mpo, eigenvector) - values[0] * eigenvector
    )
    diagnostics: dict[str, float | int] = {
        "product_basis_dimension": dimension,
        "matvec_calls": calls,
        "residual_norm": float(residual),
        "elapsed_seconds": elapsed,
        "dense_hamiltonian_materialized": False,
    }
    return float(values[0]), eigenvector, diagnostics


def _n2_ground_energy(
    basis_order: int,
    distance_length: float,
    interaction_degree: int,
    quadrature_order: int,
) -> float:
    mpo = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        2,
        basis_order,
        distance_length,
        interaction_degree,
        interaction_quadrature_order=quadrature_order,
    )
    return float(torch.linalg.eigvalsh(mpo.to_dense())[0])


def _n2_controls() -> dict[str, object]:
    basis_scan = []
    for basis_order in [6, 8, 10, 12, 16, 20]:
        energy = _n2_ground_energy(basis_order, 9.0, 32, 260)
        basis_scan.append(
            {
                "basis_order": basis_order,
                "distance_length": 9.0,
                "ground_energy": energy,
                "error_vs_independent_continuum_reference": (
                    energy - N2_CONTINUUM_REFERENCE
                ),
            }
        )

    box_scan = []
    for distance_length in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        energy = _n2_ground_energy(20, distance_length, 32, 260)
        box_scan.append(
            {
                "basis_order": 20,
                "distance_length": distance_length,
                "ground_energy": energy,
                "error_vs_independent_continuum_reference": (
                    energy - N2_CONTINUUM_REFERENCE
                ),
            }
        )

    degree_scan = []
    for degree in [8, 12, 16, 20, 24, 32]:
        energy = _n2_ground_energy(12, 9.0, degree, 220)
        degree_scan.append(
            {
                "interaction_degree": degree,
                "ground_energy": energy,
                "difference_vs_degree_32": None,
                "sampled_scalar_error": soft_coulomb_chebyshev_sampled_error(
                    9.0, degree
                ),
            }
        )
    degree_reference = degree_scan[-1]["ground_energy"]
    for point in degree_scan:
        point["difference_vs_degree_32"] = (
            point["ground_energy"] - degree_reference
        )

    quadrature_scan = []
    for quadrature_order in [48, 64, 80, 120, 160, 220]:
        energy = _n2_ground_energy(12, 9.0, 20, quadrature_order)
        quadrature_scan.append(
            {
                "quadrature_order": quadrature_order,
                "ground_energy": energy,
                "difference_vs_q220": None,
            }
        )
    quadrature_reference = quadrature_scan[-1]["ground_energy"]
    for point in quadrature_scan:
        point["difference_vs_q220"] = (
            point["ground_energy"] - quadrature_reference
        )

    return {
        "independent_continuum_reference": N2_CONTINUUM_REFERENCE,
        "reference_provenance": (
            "second-order Richardson extrapolation of an independent "
            "relative-coordinate half-line finite-difference solve"
        ),
        "basis_order_at_fixed_box": basis_scan,
        "outer_box_at_fixed_basis_order": box_scan,
        "interaction_degree_at_fixed_basis_and_quadrature": degree_scan,
        "projection_quadrature_at_fixed_basis_and_degree": quadrature_scan,
    }


def _noninteracting_basis_controls() -> dict[str, object]:
    exact = 4.5
    basis_comparison = []
    for basis_order in [2, 4, 6, 8, 10]:
        for basis_name, scale in [
            ("dirichlet_sine", 6.0),
            ("odd_hermite", 0.7),
        ]:
            hamiltonian = ordered_continuous_noninteracting_mpo(
                3,
                basis_order,
                scale,
                distance_basis=basis_name,
            ).to_dense()
            energy = float(torch.linalg.eigvalsh(hamiltonian)[0])
            basis_comparison.append(
                {
                    "basis": basis_name,
                    "basis_order": basis_order,
                    "box_or_length_scale": scale,
                    "ground_energy": energy,
                    "error_vs_exact": energy - exact,
                }
            )
    scale_scan = []
    for scale in [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
        hamiltonian = ordered_continuous_noninteracting_mpo(
            3, 8, scale, distance_basis="odd_hermite"
        ).to_dense()
        energy = float(torch.linalg.eigvalsh(hamiltonian)[0])
        scale_scan.append(
            {
                "basis_order": 8,
                "length_scale": scale,
                "ground_energy": energy,
                "error_vs_exact": energy - exact,
            }
        )
    return {
        "particles": 3,
        "exact_continuum_energy": exact,
        "basis_comparison": basis_comparison,
        "odd_hermite_length_scale_scan": scale_scan,
    }


def _n4_exact_point(
    basis_order: int,
    distance_length: float,
    interaction_degree: int,
    quadrature_order: int,
) -> dict[str, object]:
    mpo = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        4,
        basis_order,
        distance_length,
        interaction_degree,
        interaction_quadrature_order=quadrature_order,
    )
    energy, _, diagnostics = _lowest_mpo_eigenpair(mpo)
    return {
        "basis_order": basis_order,
        "distance_length": distance_length,
        "interaction_degree": interaction_degree,
        "quadrature_order": quadrature_order,
        "ground_energy": energy,
        "error_vs_exterior_D14_reference": energy - N4_EXTERIOR_D14_REFERENCE,
        **diagnostics,
    }


def _n4_galerkin_controls() -> dict[str, object]:
    basis_scan = [
        _n4_exact_point(basis_order, 4.5, 20, 160)
        for basis_order in [6, 8, 10, 12]
    ]
    box_scan = [
        _n4_exact_point(10, distance_length, 20, 160)
        for distance_length in [3.5, 4.0, 4.5, 5.0]
    ]
    degree_scan = [
        _n4_exact_point(10, 4.5, degree, 160)
        for degree in [8, 12, 16, 20, 24]
    ]
    degree_reference = degree_scan[-1]["ground_energy"]
    for point in degree_scan:
        point["difference_vs_degree_24"] = (
            point["ground_energy"] - degree_reference
        )
    return {
        "post_run_exterior_reference": N4_EXTERIOR_D14_REFERENCE,
        "reference_is_numerical_not_continuum_bound": True,
        "basis_order_at_fixed_box": basis_scan,
        "outer_box_at_fixed_basis_order": box_scan,
        "interaction_degree_at_fixed_basis": degree_scan,
    }


def _n4_mps_bond_control() -> dict[str, object]:
    from latticetn.mps import MPS

    mpo = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        4,
        10,
        4.5,
        20,
        interaction_quadrature_order=160,
    )
    exact_energy, exact_vector, lanczos = _lowest_mpo_eigenpair(mpo, seed=1620)
    exact_tensor = exact_vector.reshape((10,) * 4)
    points = []
    for bond in [1, 2, 4, 8, 16, 32, 64, 100]:
        cores, ranks, discarded_norm = particle_tensor_to_mps_tensors(
            exact_tensor, max_bond=bond
        )
        compressed = MPS.from_tensors(cores, dtype=torch.float64)
        energy = compressed.energy_with_MPO(mpo).detach()
        compressed_vector = compressed.to_dense().detach()
        fidelity = torch.vdot(exact_vector, compressed_vector).abs().square() / (
            torch.vdot(exact_vector, exact_vector).real
            * torch.vdot(compressed_vector, compressed_vector).real
        )
        points.append(
            {
                "requested_maximum_bond": bond,
                "retained_ranks": list(ranks),
                "energy": float(energy.cpu()),
                "energy_error_vs_exact_galerkin_ground": (
                    float(energy.cpu()) - exact_energy
                ),
                "fidelity_vs_exact_galerkin_ground": float(fidelity.cpu()),
                "sequential_discarded_singular_value_norm": float(
                    discarded_norm.cpu()
                ),
            }
        )
    return {
        "method": "TT-SVD compression of the independent Galerkin ground state",
        "truth_state_materialization_is_audit_only": True,
        "training_materializes_product_basis_state": False,
        "exact_galerkin_ground_energy": exact_energy,
        "lanczos": lanczos,
        "points": points,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-n4-galerkin", action="store_true", help="development-only shortcut"
    )
    parser.add_argument(
        "--skip-bond-control", action="store_true", help="development-only shortcut"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase16_ordered_continuous_controls.json"
        ),
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    record: dict[str, object] = {
        "schema_version": 1,
        "experiment": "phase16_ordered_continuous_independent_error_controls",
        "dtype": "float64",
        "antisymmetry_audit": _antisymmetry_audit(),
        "mpo_matvec_audit": _small_dense_matvec_audit(),
        "n2": _n2_controls(),
        "noninteracting_basis": _noninteracting_basis_controls(),
    }
    if not arguments.skip_n4_galerkin:
        record["n4_galerkin"] = _n4_galerkin_controls()
    if not arguments.skip_bond_control:
        record["n4_mps_bond"] = _n4_mps_bond_control()
    record["elapsed_seconds"] = time.perf_counter() - started
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
