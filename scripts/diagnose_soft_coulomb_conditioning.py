"""Diagnose finite-AGP overlap conditioning at D=8 and D=10."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from femps.algorithms import (
    canonical_pair_matrices,
    contribution_gram_spectrum,
    leave_one_out_energies,
    solve_generalized_hermitian,
)
from femps.basis import harmonic_hamiltonian
from femps.hamiltonians import agp_hamiltonian_transition_matrices, soft_coulomb_operator


def _diagnose(path: Path, dimension: int, quadrature: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    pairs = canonical_pair_matrices(payload["best_raw"])
    one_body = harmonic_hamiltonian(dimension, dtype=torch.complex128)
    interaction, _ = soft_coulomb_operator(
        dimension,
        quadrature_order=quadrature,
        relative_threshold=1e-14,
    )
    overlap, hamiltonian = agp_hamiltonian_transition_matrices(
        pairs, 2, one_body, interaction
    )
    solved = solve_generalized_hermitian(hamiltonian, overlap)
    diagonal = torch.diagonal(overlap).real
    normalized_overlap = torch.abs(overlap) / torch.sqrt(
        diagonal[:, None] * diagonal[None, :]
    )
    pair_vectors = pairs.reshape(pairs.shape[0], -1)
    pair_similarity = torch.abs(pair_vectors.conj() @ pair_vectors.transpose(0, 1))
    leave_out = leave_one_out_energies(hamiltonian, overlap)
    return {
        "D": dimension,
        "Q": quadrature,
        "K": pairs.shape[0],
        "energy": float(solved.energy),
        "balanced_overlap_eigenvalues": [
            float(x) for x in solved.overlap_eigenvalues
        ],
        "balanced_overlap_condition_number": solved.retained_condition_number,
        "raw_overlap_eigenvalues": [
            float(x) for x in solved.raw_overlap_eigenvalues
        ],
        "raw_overlap_condition_number": solved.raw_overlap_condition_number,
        "normalized_state_overlap_magnitudes": normalized_overlap.tolist(),
        "pair_matrix_similarity_magnitudes": pair_similarity.tolist(),
        "amplitude_magnitudes": [float(x) for x in torch.abs(solved.amplitudes)],
        "contribution_gram_spectrum": [
            float(x) for x in contribution_gram_spectrum(overlap, solved.amplitudes)
        ],
        "leave_one_out_energies": [float(x) for x in leave_out],
        "leave_one_out_penalties": [float(x - solved.energy) for x in leave_out],
        "generalized_residual_norm": float(solved.residual_norm),
    }


def main() -> None:
    result = {
        "schema_version": 1,
        "experiment": "soft_coulomb_finite_agp_conditioning_diagnostics",
        "diagnostic_not_entanglement_spectrum": True,
        "points": [
            _diagnose(
                Path(
                    "docs/experiments/results/soft_coulomb_n4_hierarchy_checkpoints/"
                    "k4_joint_checkpoint.pt"
                ),
                8,
                96,
            ),
            _diagnose(
                Path(
                    "docs/experiments/results/soft_coulomb_n4_d10_hierarchy_checkpoints/"
                    "k4_joint_checkpoint.pt"
                ),
                10,
                128,
            ),
        ],
    }
    output = Path("docs/experiments/results/soft_coulomb_conditioning.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for point in result["points"]:
        print(
            f"D={point['D']} raw_condition="
            f"{point['raw_overlap_condition_number']:.3f} balanced_condition="
            f"{point['balanced_overlap_condition_number']:.3f} "
            f"spectrum_min={min(point['contribution_gram_spectrum']):.3e} "
            f"min_leave_out={min(point['leave_one_out_penalties']):.3e}"
        )


if __name__ == "__main__":
    main()
