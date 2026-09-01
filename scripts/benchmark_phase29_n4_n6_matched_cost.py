"""Measure matched N=4 versus N=6 diagonal-transition cost at D10,K4,L19."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from femps.algorithms import canonical_slater_orbitals, load_diagonal_path_checkpoint
from femps.benchmarks import ProcessRSSMonitor
from femps.exterior import (
    diagonal_path_hamiltonian_matrices,
    diagonal_path_structural_counts,
    diagonal_path_transition_diagnostics,
)
from femps.hamiltonians import harmonic_pair_hamiltonian, soft_coulomb_operator


DIMENSION = 10
TERMS = 4
QUADRATURE = 128
FACTORIZATION_TOLERANCE = 1e-11
VALUE_TOLERANCE = 1e-10


def _evaluate(
    orbitals: torch.Tensor,
    one_body: torch.Tensor,
    interaction,
    *,
    algorithm: str,
    backward: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = orbitals.detach().clone().requires_grad_(backward)
    overlap, hamiltonian = diagonal_path_hamiltonian_matrices(
        state,
        one_body,
        two_body_left=interaction.left,
        two_body_right=interaction.right,
        two_body_weights=interaction.weights,
        transition_algorithm=algorithm,
    )
    scalar = overlap.abs().square().sum() + hamiltonian.abs().square().sum()
    if backward:
        scalar.backward()
        if state.grad is None:
            raise RuntimeError("reverse-mode timing did not produce a gradient")
    return overlap.detach(), hamiltonian.detach()


def _timings(
    orbitals: torch.Tensor,
    one_body: torch.Tensor,
    interaction,
    *,
    algorithm: str,
    backward: bool,
    repeats: int,
) -> dict:
    _evaluate(
        orbitals,
        one_body,
        interaction,
        algorithm=algorithm,
        backward=backward,
    )
    samples = []
    with ProcessRSSMonitor() as monitor:
        for _ in range(repeats):
            started = time.perf_counter()
            _evaluate(
                orbitals,
                one_body,
                interaction,
                algorithm=algorithm,
                backward=backward,
            )
            samples.append(time.perf_counter() - started)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "cpu_memory": monitor.record().as_dict(),
    }


def _point(
    *,
    particles: int,
    checkpoint: Path,
    state_record: dict,
    dense_ci_record: dict,
    one_body: torch.Tensor,
    interaction,
    repeats: int,
) -> dict:
    payload = load_diagonal_path_checkpoint(checkpoint)
    orbitals = canonical_slater_orbitals(payload["best_raw"])
    auto_overlap, auto_hamiltonian = _evaluate(
        orbitals, one_body, interaction, algorithm="auto", backward=False
    )
    minor_overlap, minor_hamiltonian = _evaluate(
        orbitals, one_body, interaction, algorithm="minor", backward=False
    )
    modes = {}
    for algorithm in ("auto", "minor"):
        for backward in (False, True):
            label = f"{algorithm}_{'forward_backward' if backward else 'forward'}"
            modes[label] = _timings(
                orbitals,
                one_body,
                interaction,
                algorithm=algorithm,
                backward=backward,
                repeats=repeats,
            )
    return {
        "N": particles,
        "D": DIMENSION,
        "K": TERMS,
        "L": interaction.rank,
        "checkpoint": str(checkpoint),
        "auto_minor_overlap_max_absolute_difference": float(
            torch.max(torch.abs(auto_overlap - minor_overlap))
        ),
        "auto_minor_hamiltonian_max_absolute_difference": float(
            torch.max(torch.abs(auto_hamiltonian - minor_hamiltonian))
        ),
        "transition_diagnostics": diagonal_path_transition_diagnostics(orbitals),
        "structural_counts": diagonal_path_structural_counts(
            particles, DIMENSION, TERMS, interaction.rank
        ),
        "femps_ordinary_particle_tt_ranks": state_record["ordinary_particle_tt_ranks"],
        "dense_ci_ordinary_particle_tt_ranks": dense_ci_record[
            "ordinary_particle_tt_ranks"
        ],
        "exterior_dimension": dense_ci_record["exterior_dimension"],
        "modes": modes,
        "forward_speedup_minor_over_auto": (
            modes["minor_forward"]["median_seconds"]
            / modes["auto_forward"]["median_seconds"]
        ),
        "forward_backward_speedup_minor_over_auto": (
            modes["minor_forward_backward"]["median_seconds"]
            / modes["auto_forward_backward"]["median_seconds"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--n4-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/phase28_soft_coulomb_basis_extension/N4_D10_K4_seed17.pt"
        ),
    )
    parser.add_argument(
        "--n6-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/phase29_n6_multiseed_stability/N6_D10_K4_seed31.pt"
        ),
    )
    parser.add_argument(
        "--n4-artifact",
        type=Path,
        default=Path(
            "docs/experiments/results/phase28_soft_coulomb_basis_extension.json"
        ),
    )
    parser.add_argument(
        "--n6-artifact",
        type=Path,
        default=Path(
            "docs/experiments/results/phase29_n6_multiseed_stability.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/phase29_n4_n6_matched_cost.json"),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    for checkpoint in (args.n4_checkpoint, args.n6_checkpoint):
        if not checkpoint.exists():
            raise ValueError(f"missing accepted checkpoint: {checkpoint}")

    n4_artifact = json.loads(args.n4_artifact.read_text(encoding="utf-8"))
    n6_artifact = json.loads(args.n6_artifact.read_text(encoding="utf-8"))
    n4_state = n4_artifact["extension_points"][0]
    n4_truth = n4_artifact["operator_and_truth_audits"][0]["dense_ci"]
    n6_state = n6_artifact["points"][0]
    n6_truth = n6_artifact["dense_ci_audit"]

    one_body = harmonic_pair_hamiltonian(
        DIMENSION, kappa=0.0, dtype=torch.complex128, device="cpu"
    )[0]
    interaction, diagnostics = soft_coulomb_operator(
        DIMENSION,
        quadrature_order=QUADRATURE,
        coupling=1.0,
        softening=1.0,
        relative_threshold=1e-13,
        factorization_backend="physical",
        dtype=torch.complex128,
        device="cpu",
    )
    points = [
        _point(
            particles=4,
            checkpoint=args.n4_checkpoint,
            state_record=n4_state,
            dense_ci_record=n4_truth,
            one_body=one_body,
            interaction=interaction,
            repeats=args.repeats,
        ),
        _point(
            particles=6,
            checkpoint=args.n6_checkpoint,
            state_record=n6_state,
            dense_ci_record=n6_truth,
            one_body=one_body,
            interaction=interaction,
            repeats=args.repeats,
        ),
    ]
    n4, n6 = points
    ratios = {
        "stored_orbital_scalars_N6_over_N4": (
            n6["structural_counts"]["stored_orbital_scalars"]
            / n4["structural_counts"]["stored_orbital_scalars"]
        ),
        "one_body_determinants_N6_over_N4": (
            n6["structural_counts"]["one_body_determinants"]
            / n4["structural_counts"]["one_body_determinants"]
        ),
        "two_body_determinants_N6_over_N4": (
            n6["structural_counts"]["two_body_determinants"]
            / n4["structural_counts"]["two_body_determinants"]
        ),
    }
    for mode in (
        "auto_forward",
        "auto_forward_backward",
        "minor_forward",
        "minor_forward_backward",
    ):
        ratios[f"{mode}_median_N6_over_N4"] = (
            n6["modes"][mode]["median_seconds"]
            / n4["modes"][mode]["median_seconds"]
        )
    point_pass = [
        bool(
            point["auto_minor_overlap_max_absolute_difference"] <= VALUE_TOLERANCE
            and point["auto_minor_hamiltonian_max_absolute_difference"]
            <= VALUE_TOLERANCE
            and point["structural_counts"]["enumerated_virtual_paths"] == 0
            and point["structural_counts"]["materialized_particle_coefficients"]
            == 0
            and all(
                mode["median_seconds"] > 0 and mode["cpu_memory"]["peak_rss_bytes"] > 0
                for mode in point["modes"].values()
            )
        )
        for point in points
    ]
    operator_pass = bool(
        diagnostics.factorization_backend == "physical_operator_svd"
        and diagnostics.retained_rank == 19
        and diagnostics.dense_relative_factorization_error
        <= FACTORIZATION_TOLERANCE
    )
    accepted = all(point_pass) and operator_pass
    artifact = {
        "schema_version": 1,
        "experiment": "phase29_matched_N4_N6_diagonal_transition_cost",
        "evidence_level": "numerical",
        "scientific_boundary": "matched kernel timing at fixed D,K,L; not asymptotic scaling",
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "repeats": args.repeats,
        "operator_audit": {
            "backend": diagnostics.factorization_backend,
            "rank": diagnostics.retained_rank,
            "dense_relative_factorization_error": (
                diagnostics.dense_relative_factorization_error
            ),
        },
        "points": points,
        "N6_over_N4_ratios": ratios,
        "acceptance": {
            "per_point_pass": point_pass,
            "operator_pass": operator_pass,
            "matched_cost_audit_pass": accepted,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "matched_cost_audit_pass": accepted}, indent=2))


if __name__ == "__main__":
    main()
