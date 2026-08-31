"""Polynomial-energy refinement of finite-AGP E4 states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from femps.devices import resolve_device
from femps.exterior import agp_exterior_coefficients, agp_tensor, particle_tt_ranks
from femps.hamiltonians import (
    agp_sum_energy,
    antisymmetric_many_body_hamiltonian,
    harmonic_pair_hamiltonian,
)


def _random_complex(
    shape: tuple[int, ...], seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imaginary).to(device)


def _project(raw: torch.Tensor, amplitudes: torch.Tensor, pairs: int) -> None:
    with torch.no_grad():
        pair_matrices = raw - raw.transpose(1, 2)
        norms = torch.linalg.vector_norm(pair_matrices, dim=(1, 2)).clamp_min(1e-14)
        raw.div_(norms[:, None, None])
        amplitudes.mul_(norms.pow(pairs))
        amplitude_norm = torch.linalg.vector_norm(amplitudes)
        if amplitude_norm <= 1e-14:
            raise RuntimeError("finite-AGP amplitudes collapsed to zero")
        amplitudes.div_(amplitude_norm)


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=8)
    parser.add_argument("--kappa", type=float, default=0.35)
    parser.add_argument("--length", type=int, default=2)
    parser.add_argument(
        "--initializations", nargs="+", default=["oracle", "random"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--oracle-checkpoint",
        type=Path,
        default=Path(
            "docs/experiments/results/fermion_e4_agp_rank_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/results/fermion_e4_k2_energy_refinement.json"
        ),
    )
    args = parser.parse_args()
    if any(mode not in {"oracle", "random"} for mode in args.initializations):
        raise ValueError("initializations must contain only oracle and/or random")
    device = resolve_device(args.device)
    one_body_cpu, interaction_cpu = harmonic_pair_hamiltonian(
        args.basis_order, kappa=args.kappa, device="cpu"
    )
    exact_hamiltonian = antisymmetric_many_body_hamiltonian(
        one_body_cpu, 4, interaction_cpu
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(exact_hamiltonian)
    finite_energy = float(eigenvalues[0].real)
    exact_coefficients = eigenvectors[:, 0]
    one_body, interaction = harmonic_pair_hamiltonian(
        args.basis_order, kappa=args.kappa, device=device
    )
    oracle = None
    if "oracle" in args.initializations:
        payload = torch.load(
            args.oracle_checkpoint, map_location="cpu", weights_only=False
        )
        if args.length not in payload["states"]:
            raise ValueError("requested length is absent from oracle checkpoint")
        oracle = payload["states"][args.length]

    result = {
        "schema_version": 1,
        "experiment": "fermion_e4_finite_agp_polynomial_energy_refinement",
        "D": args.basis_order,
        "N": 4,
        "K": args.length,
        "kappa": args.kappa,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "final_learning_rate": args.final_learning_rate,
        "finite_basis_reference_energy": finite_energy,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "runs": [],
    }
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    completed = {
        (run["initialization"], run["seed"]) for run in result["runs"]
    }

    for initialization in args.initializations:
        run_seeds = [int(oracle["selected_seed"])] if initialization == "oracle" else args.seeds
        for seed in run_seeds:
            if (initialization, seed) in completed:
                print(f"skip initialization={initialization} seed={seed}")
                continue
            if initialization == "oracle":
                assert oracle is not None
                initial_pairs = oracle["pair_matrices"].to(torch.complex128).to(device)
                initial_amplitudes = oracle["amplitudes"].to(torch.complex128).to(device)
                initial_raw = 0.5 * initial_pairs
            else:
                initial_raw = _random_complex(
                    (args.length, args.basis_order, args.basis_order),
                    seed,
                    device,
                )
                initial_amplitudes = _random_complex(
                    (args.length,), seed + 10000, device
                )
            raw = torch.nn.Parameter(initial_raw)
            amplitudes = torch.nn.Parameter(initial_amplitudes)
            _project(raw, amplitudes, pairs=2)
            optimizer = torch.optim.Adam(
                [raw, amplitudes], lr=args.learning_rate
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.steps,
                eta_min=args.final_learning_rate,
            )

            def energy() -> torch.Tensor:
                return agp_sum_energy(
                    raw - raw.transpose(1, 2),
                    amplitudes,
                    2,
                    one_body,
                    interaction,
                )

            interval = max(1, args.steps // 20)
            history = [
                {
                    "step": 0,
                    "energy": float(energy().detach().cpu()),
                    "learning_rate": args.learning_rate,
                }
            ]
            best_energy = history[0]["energy"]
            best_raw = raw.detach().clone()
            best_amplitudes = amplitudes.detach().clone()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for step in range(1, args.steps + 1):
                optimizer.zero_grad()
                loss = energy()
                loss.backward()
                optimizer.step()
                _project(raw, amplitudes, pairs=2)
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
                        best_amplitudes = amplitudes.detach().clone()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            terminal_energy = float(energy().detach().cpu())
            with torch.no_grad():
                raw.copy_(best_raw)
                amplitudes.copy_(best_amplitudes)
            final_energy = float(energy().detach().cpu())
            final_pairs = (raw - raw.transpose(1, 2)).detach().cpu()
            final_amplitudes = amplitudes.detach().cpu()
            coefficients = sum(
                final_amplitudes[term]
                * agp_exterior_coefficients(final_pairs[term], 2)
                for term in range(args.length)
            )
            coefficient_norm = torch.vdot(coefficients, coefficients).real
            explicit_energy = float(
                (
                    torch.vdot(coefficients, exact_hamiltonian @ coefficients)
                    / coefficient_norm
                ).real
            )
            fidelity = float(
                (
                    torch.abs(torch.vdot(exact_coefficients, coefficients)) ** 2
                    / coefficient_norm
                ).real
            )
            particle_state = sum(
                final_amplitudes[term] * agp_tensor(final_pairs[term], 2)
                for term in range(args.length)
            )
            run = {
                "initialization": initialization,
                "seed": seed,
                "initial_energy": history[0]["energy"],
                "terminal_energy_before_best_restore": terminal_energy,
                "final_energy": final_energy,
                "explicit_exterior_energy": explicit_energy,
                "polynomial_explicit_absolute_difference": abs(
                    final_energy - explicit_energy
                ),
                "error_vs_finite_basis": final_energy - finite_energy,
                "finite_basis_ground_fidelity": fidelity,
                "ordinary_particle_tt_ranks": list(
                    particle_tt_ranks(particle_state)
                ),
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
                f"initialization={initialization} seed={seed} "
                f"E={final_energy:.12f} finite_error={run['error_vs_finite_basis']:.3e}"
            )


if __name__ == "__main__":
    main()
