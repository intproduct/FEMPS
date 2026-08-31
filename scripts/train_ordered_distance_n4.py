"""Blind multi-seed Gate C optimization followed by a small truth audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from femps.algorithms.ordered_distance_training import (
    OrderedDistanceTrainingConfig,
    train_ordered_distance_mps,
)
from femps.baselines.ordered_distance_mpo import gap_charge_projector_mpo
from femps.devices import resolve_device
from femps.ordered_distance import gap_configurations, gap_hamiltonian


def _flat_gap_indices(
    grid_points: int, particles: int, gap_cutoff: int
) -> torch.Tensor:
    sites = particles + 1
    local_dimension = gap_cutoff + 1
    return torch.tensor(
        [
            sum(
                value * local_dimension ** (sites - 1 - site)
                for site, value in enumerate(gaps)
            )
            for gaps in gap_configurations(
                grid_points, particles, gap_cutoff=gap_cutoff
            )
        ],
        dtype=torch.long,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-points", type=int, default=8)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=0.7)
    parser.add_argument("--gap-cutoff", type=int, default=4)
    parser.add_argument("--multiplicity", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seeds", nargs="+", type=int, default=[701, 1701, 2701])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--energy-tolerance", type=float, default=5e-5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/phase15_ordered_distance_training.json"
        ),
    )
    arguments = parser.parse_args()
    device = resolve_device(arguments.device)
    holes = arguments.grid_points - arguments.particles
    if arguments.gap_cutoff * (arguments.particles + 1) < holes:
        raise ValueError("the local gap cutoff cannot carry the finite-box charge")

    # These runs know only the native MPS/MPO Rayleigh quotient.  Exact
    # diagonalization and dense state gathering happen below, after every run.
    run_records = []
    trained_cores = []
    for seed in arguments.seeds:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        config = OrderedDistanceTrainingConfig(
            grid_points=arguments.grid_points,
            particles=arguments.particles,
            spacing=arguments.spacing,
            gap_cutoff=arguments.gap_cutoff,
            multiplicity_per_charge=arguments.multiplicity,
            steps=arguments.steps,
            learning_rate=arguments.learning_rate,
            seed=seed,
            device=str(device),
        )
        mps, diagnostics = train_ordered_distance_mps(config)
        projector = gap_charge_projector_mpo(
            arguments.grid_points,
            arguments.particles,
            gap_cutoff=arguments.gap_cutoff,
            dtype=config.dtype,
            device=device,
        )
        charge_weight = (
            mps._expect_MPO(projector) / mps.overlap(mps)
        ).real
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
        trained_cores.append([core.detach().cpu().clone() for core in mps.tensors])
        run_records.append(
            {
                "seed": seed,
                **diagnostics,
                "charge_weight": float(charge_weight.detach().cpu()),
                "seconds": seconds,
                "peak_cuda_memory_bytes": peak_memory,
            }
        )

    # Independent, deliberately small truth audit.  Nothing above depends on
    # this matrix, its eigenvector, or a D**N state materialization.
    truth_hamiltonian = gap_hamiltonian(
        arguments.grid_points,
        arguments.particles,
        arguments.spacing,
        gap_cutoff=arguments.gap_cutoff,
        soft_coulomb=True,
    )
    truth_eigenvalues, truth_eigenvectors = torch.linalg.eigh(truth_hamiltonian)
    truth_energy = float(truth_eigenvalues[0])
    truth_state = truth_eigenvectors[:, 0]
    flat_indices = _flat_gap_indices(
        arguments.grid_points,
        arguments.particles,
        arguments.gap_cutoff,
    )
    from latticetn.mps import MPS

    for record, cores in zip(run_records, trained_cores, strict=True):
        audit_mps = MPS.from_tensors(
            cores,
            dtype=cores[0].dtype,
            device="cpu",
            requires_grad=False,
        )
        dense_sector = audit_mps.to_dense()[flat_indices]
        dense_sector = dense_sector / torch.linalg.vector_norm(dense_sector)
        fidelity = torch.abs(torch.vdot(truth_state, dense_sector)) ** 2
        energy_error = float(record["best_energy"] - truth_energy)
        record["energy_error_vs_post_training_truth"] = energy_error
        record["ground_state_fidelity"] = float(fidelity)
        record["passes_energy_tolerance"] = (
            abs(energy_error) <= arguments.energy_tolerance
        )

    gate_pass = all(
        record["passes_energy_tolerance"]
        and abs(record["charge_weight"] - 1.0) <= 2e-12
        and record["max_forbidden_parameter"] == 0.0
        for record in run_records
    )
    result = {
        "schema_version": 1,
        "experiment": "phase15_ordered_distance_blind_training",
        "evidence_level": "native_blind_training_then_small_exact_truth_audit",
        "grid_points": arguments.grid_points,
        "particles": arguments.particles,
        "spacing": arguments.spacing,
        "gap_cutoff": arguments.gap_cutoff,
        "local_dimension": arguments.gap_cutoff + 1,
        "multiplicity_per_charge": arguments.multiplicity,
        "steps": arguments.steps,
        "learning_rate": arguments.learning_rate,
        "energy_tolerance": arguments.energy_tolerance,
        "device": str(device),
        "dtype": "float64",
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "compute_capability": (
            list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        ),
        "training_materializes_D_power_N": False,
        "post_training_truth_audit_materializes_small_local_gap_tensor": True,
        "truth_sector_dimension": truth_hamiltonian.shape[0],
        "truth_ground_energy": truth_energy,
        "runs": run_records,
        "all_seed_gate_pass": gate_pass,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
