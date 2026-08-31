"""Blind single-AGP Blackwell scan for interacting four-fermion E4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from femps.devices import resolve_device
from femps.exterior import (
    agp_exterior_coefficients,
    agp_tensor,
    antisymmetry_residual,
    bivector_decomposition_length,
    particle_tt_ranks,
)
from femps.hamiltonians import (
    agp_energy,
    antisymmetric_many_body_hamiltonian,
    exact_interacting_harmonic_fermion_energy,
    harmonic_pair_hamiltonian,
)


def _random_raw(dimension: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(
        dimension, dimension, generator=generator, dtype=torch.float64
    )
    raw = torch.complex(real, imaginary).to(device)
    return raw / torch.linalg.vector_norm(raw - raw.transpose(0, 1))


def _write(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=8)
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.1, 0.35, 0.8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--record-points", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/fermion_e4_single_agp_sweep.json"),
    )
    args = parser.parse_args()
    if args.basis_order < 4 or args.steps < 1:
        raise ValueError("require D >= 4 and positive steps")
    if not 0 < args.final_learning_rate <= args.learning_rate:
        raise ValueError("invalid learning-rate interval")
    device = resolve_device(args.device)
    result = {
        "schema_version": 1,
        "experiment": "fermion_e4_blind_single_agp_sweep",
        "particles": 4,
        "pairs": 2,
        "basis_order": args.basis_order,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "final_learning_rate": args.final_learning_rate,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "runs": [],
    }
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    completed = {(run["kappa"], run["seed"]) for run in result["runs"]}

    for kappa in args.kappas:
        one_body_cpu, interaction_cpu = harmonic_pair_hamiltonian(
            args.basis_order, kappa=kappa, device="cpu"
        )
        exact_hamiltonian = antisymmetric_many_body_hamiltonian(
            one_body_cpu, 4, interaction_cpu
        )
        exact_values, exact_vectors = torch.linalg.eigh(exact_hamiltonian)
        finite_energy = float(exact_values[0].real)
        exact_coefficients = exact_vectors[:, 0]
        continuum_energy = exact_interacting_harmonic_fermion_energy(
            4, kappa=kappa
        )
        one_body = one_body_cpu.to(device)
        interaction = type(interaction_cpu)(
            interaction_cpu.left.to(device),
            interaction_cpu.right.to(device),
            interaction_cpu.weights.to(device),
        )
        for seed in args.seeds:
            if (kappa, seed) in completed:
                print(f"skip kappa={kappa:g} seed={seed}")
                continue
            raw = torch.nn.Parameter(_random_raw(args.basis_order, seed, device))
            optimizer = torch.optim.Adam([raw], lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.steps,
                eta_min=args.final_learning_rate,
            )

            def energy() -> torch.Tensor:
                return agp_energy(
                    raw - raw.transpose(0, 1), 2, one_body, interaction
                )

            interval = max(1, args.steps // args.record_points)
            history = [
                {
                    "step": 0,
                    "energy": float(energy().detach().cpu()),
                    "learning_rate": args.learning_rate,
                }
            ]
            best_energy = history[0]["energy"]
            best_raw = raw.detach().clone()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for step in range(1, args.steps + 1):
                optimizer.zero_grad()
                loss = energy()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    raw.div_(torch.linalg.vector_norm(raw - raw.transpose(0, 1)))
                scheduler.step()
                if step % interval == 0 or step == args.steps:
                    observed = float(energy().detach().cpu())
                    history.append(
                        {
                            "step": step,
                            "energy": observed,
                            "learning_rate": scheduler.get_last_lr()[0],
                        }
                    )
                    if observed < best_energy:
                        best_energy = observed
                        best_raw = raw.detach().clone()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            terminal_energy = float(energy().detach().cpu())
            with torch.no_grad():
                raw.copy_(best_raw)
            final_energy = float(energy().detach().cpu())
            final_pair = (raw - raw.transpose(0, 1)).detach().cpu()
            coefficients = agp_exterior_coefficients(final_pair, 2)
            coefficient_norm = torch.vdot(coefficients, coefficients).real
            explicit_energy = float(
                (
                    torch.vdot(
                        coefficients,
                        exact_hamiltonian @ coefficients,
                    )
                    / coefficient_norm
                ).real
            )
            fidelity = float(
                (
                    torch.abs(torch.vdot(exact_coefficients, coefficients)) ** 2
                    / coefficient_norm
                ).real
            )
            state = agp_tensor(final_pair, 2)
            tolerance = (
                coefficients.numel()
                * torch.finfo(coefficients.real.dtype).eps
                * torch.max(torch.abs(coefficients))
            )
            run = {
                "kappa": kappa,
                "seed": seed,
                "continuum_reference_energy": continuum_energy,
                "finite_basis_reference_energy": finite_energy,
                "basis_error_vs_continuum": abs(finite_energy - continuum_energy),
                "initial_energy": history[0]["energy"],
                "terminal_energy_before_best_restore": terminal_energy,
                "final_energy": final_energy,
                "explicit_exterior_energy": explicit_energy,
                "polynomial_explicit_absolute_difference": abs(
                    final_energy - explicit_energy
                ),
                "error_vs_finite_basis": final_energy - finite_energy,
                "error_vs_continuum": final_energy - continuum_energy,
                "finite_basis_ground_fidelity": fidelity,
                "pair_matrix_channel_length": bivector_decomposition_length(final_pair),
                "exterior_support_dimension": int(
                    torch.count_nonzero(torch.abs(coefficients) > tolerance)
                ),
                "exterior_dimension": math.comb(args.basis_order, 4),
                "ordinary_particle_tt_ranks": list(particle_tt_ranks(state)),
                "antisymmetry_residual": float(antisymmetry_residual(state)),
                "history": history,
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None
                ),
            }
            if run["error_vs_finite_basis"] < -1e-8:
                raise RuntimeError("variational energy fell below finite-basis truth")
            result["runs"].append(run)
            _write(args.output, result)
            print(
                f"kappa={kappa:g} seed={seed} E={final_energy:.12f} "
                f"finite_error={run['error_vs_finite_basis']:.3e}"
            )


if __name__ == "__main__":
    main()
