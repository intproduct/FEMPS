"""Blind and resumable even-N single-AGP soft-Coulomb optimization."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import time

import torch

from femps.basis import harmonic_hamiltonian
from femps.devices import resolve_device
from femps.exterior import agp_exterior_coefficients
from femps.hamiltonians import (
    agp_energy,
    antisymmetric_many_body_hamiltonian,
    soft_coulomb_operator,
)


def _project_raw(raw: torch.Tensor) -> None:
    with torch.no_grad():
        skew = raw - raw.transpose(0, 1)
        norm = torch.linalg.vector_norm(skew)
        if norm == 0:
            raise ValueError("pair matrix became zero")
        raw.copy_(0.5 * skew / norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--basis-order", type=int, default=12)
    parser.add_argument("--quadrature-order", type=int, default=128)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--softening", type=float, default=1.0)
    parser.add_argument("--relative-threshold", type=float, default=1e-14)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps-this-call", type=int)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("docs/experiments/results/soft_coulomb_n2_checkpoint.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/soft_coulomb_n2.json"),
    )
    args = parser.parse_args()
    if args.particles < 2 or args.particles % 2 or args.particles > args.basis_order:
        raise ValueError("require an even particle count with 2 <= N <= D")
    pairs = args.particles // 2
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume", "max_steps_this_call", "checkpoint", "output"}
    }
    device = resolve_device(args.device)
    one_body = harmonic_hamiltonian(
        args.basis_order, dtype=torch.complex128, device=device
    )
    interaction, diagnostics = soft_coulomb_operator(
        args.basis_order,
        quadrature_order=args.quadrature_order,
        coupling=args.coupling,
        softening=args.softening,
        relative_threshold=args.relative_threshold,
        device=device,
    )

    resumed = args.resume and args.checkpoint.exists()
    if resumed:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if payload["config"] != config:
            raise ValueError("checkpoint configuration does not match")
        raw = torch.nn.Parameter(payload["raw"].to(device))
        best_raw = payload["best_raw"].to(device)
        best_energy = float(payload["best_energy"])
        start_step = int(payload["step"])
        history = list(payload["history"])
    else:
        generator = torch.Generator().manual_seed(args.seed)
        real = torch.randn(
            args.basis_order,
            args.basis_order,
            generator=generator,
            dtype=torch.float64,
        )
        imaginary = torch.randn(
            args.basis_order,
            args.basis_order,
            generator=generator,
            dtype=torch.float64,
        )
        raw = torch.nn.Parameter(torch.complex(real, imaginary).to(device))
        _project_raw(raw)
        best_raw = raw.detach().clone()
        start_step = 0
        history = []

    optimizer = torch.optim.Adam([raw], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-5
    )
    if resumed:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])

    def energy() -> torch.Tensor:
        return agp_energy(raw - raw.transpose(0, 1), pairs, one_body, interaction)

    if not history:
        initial_energy = float(energy().detach().cpu())
        best_energy = initial_energy
        history.append({"step": 0, "energy": initial_energy})
    stop_step = args.steps
    if args.max_steps_this_call is not None:
        stop_step = min(stop_step, start_step + args.max_steps_this_call)
    record_interval = max(1, args.steps // 20)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(start_step + 1, stop_step + 1):
        optimizer.zero_grad()
        loss = energy()
        loss.backward()
        optimizer.step()
        _project_raw(raw)
        scheduler.step()
        current = float(energy().detach().cpu())
        if current < best_energy:
            best_energy = current
            best_raw = raw.detach().clone()
        if step % record_interval == 0 or step == stop_step:
            history.append(
                {
                    "step": step,
                    "energy": current,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )
        if step % 100 == 0 or step == stop_step:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "config": config,
                    "step": step,
                    "raw": raw.detach().cpu(),
                    "best_raw": best_raw.detach().cpu(),
                    "best_energy": best_energy,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                },
                args.checkpoint,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    terminal_energy = float(energy().detach().cpu())
    with torch.no_grad():
        raw.copy_(best_raw)
    final_pair = (raw - raw.transpose(0, 1)).detach().cpu()
    final_energy = float(energy().detach().cpu())

    one_body_cpu = harmonic_hamiltonian(args.basis_order, dtype=torch.complex128)
    interaction_cpu, _ = soft_coulomb_operator(
        args.basis_order,
        quadrature_order=args.quadrature_order,
        coupling=args.coupling,
        softening=args.softening,
        relative_threshold=args.relative_threshold,
    )
    truth_hamiltonian = antisymmetric_many_body_hamiltonian(
        one_body_cpu, args.particles, interaction_cpu
    )
    truth_values, truth_vectors = torch.linalg.eigh(truth_hamiltonian)
    coefficients = agp_exterior_coefficients(final_pair, pairs)
    norm = torch.vdot(coefficients, coefficients).real
    explicit_energy = float(
        (torch.vdot(coefficients, truth_hamiltonian @ coefficients) / norm).real
    )
    fidelity = float(
        (torch.abs(torch.vdot(truth_vectors[:, 0], coefficients)) ** 2 / norm).real
    )
    result = {
        "schema_version": 1,
        "experiment": f"soft_coulomb_n{args.particles}_blind_single_agp",
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "resumed_this_call": resumed,
        "completed": stop_step == args.steps,
        "completed_steps": stop_step,
        "initial_energy": history[0]["energy"],
        "terminal_energy_before_best_restore": terminal_energy,
        "final_energy": final_energy,
        "explicit_exterior_energy": explicit_energy,
        "polynomial_exterior_absolute_difference": abs(final_energy - explicit_energy),
        "finite_basis_reference_energy": float(truth_values[0].real),
        "error_vs_finite_basis": final_energy - float(truth_values[0].real),
        "finite_basis_ground_fidelity": fidelity,
        "operator_diagnostics": asdict(diagnostics),
        "history": history,
        "elapsed_seconds_this_call": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"steps={stop_step} resumed={resumed} E={final_energy:.12f} "
        f"finite_error={result['error_vs_finite_basis']:.3e}"
    )


if __name__ == "__main__":
    main()
