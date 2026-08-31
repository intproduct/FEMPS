"""Reproduce the E3 four-fermion noninteracting representation benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch

from femps.basis import harmonic_hamiltonian
from femps.devices import resolve_device
from femps.exterior import (
    agp_femps_cores,
    agp_tensor,
    antisymmetry_residual,
    best_rank_error,
    materialize_femps_matrix,
    normalized_slater_from_minors,
    particle_schmidt_spectrum,
    particle_tt_ranks,
    slater_flat_spectrum,
    slater_sum_cores,
)
from femps.hamiltonians import (
    agp_energy,
    antisymmetric_many_body_hamiltonian,
    exact_noninteracting_fermion_energy,
)


@dataclass(frozen=True, slots=True)
class E3Config:
    basis_order: int = 8
    steps: int = 600
    learning_rate: float = 3e-2
    final_learning_rate: float = 1e-5
    seed: int = 17
    device: str = "auto"
    record_points: int = 20

    def validate(self) -> None:
        if self.basis_order < 4:
            raise ValueError("E3 requires basis_order >= 4")
        if self.steps < 1 or self.record_points < 1:
            raise ValueError("steps and record_points must be positive")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError("require 0 < final_learning_rate <= learning_rate")


def _ground_constructions(
    dimension: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orbitals = torch.eye(dimension, dtype=torch.complex128)[:, :4]
    slater = normalized_slater_from_minors(orbitals)
    slater_femps = materialize_femps_matrix(slater_sum_cores(orbitals.unsqueeze(0)))
    left = orbitals[:, (0, 2)].transpose(0, 1)
    right = orbitals[:, (1, 3)].transpose(0, 1)
    weights = torch.ones(2, dtype=torch.complex128)
    agp_femps = materialize_femps_matrix(
        agp_femps_cores(left, right, pairs=2, weights=weights)
    )
    pair_matrix = left.transpose(0, 1) @ right - right.transpose(0, 1) @ left
    return pair_matrix, slater, slater_femps, agp_femps


def _random_raw(dimension: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(
        dimension, dimension, generator=generator, dtype=torch.float64
    )
    raw = torch.complex(real, imaginary).to(device)
    return raw / torch.linalg.vector_norm(raw - raw.transpose(0, 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-order", type=int, default=8)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--record-points", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/fermion_e3.json"),
    )
    args = parser.parse_args()
    config = E3Config(
        basis_order=args.basis_order,
        steps=args.steps,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        seed=args.seed,
        device=args.device,
        record_points=args.record_points,
    )
    config.validate()
    device = resolve_device(config.device)
    particles = 4
    pairs = 2
    reference_energy = exact_noninteracting_fermion_energy(particles)
    one_body_cpu = harmonic_hamiltonian(
        config.basis_order, dtype=torch.complex128, device="cpu"
    )
    one_body = one_body_cpu.to(device)
    pair_ground, slater, slater_femps, agp_femps = _ground_constructions(
        config.basis_order
    )
    agp_ground = agp_tensor(pair_ground, pairs)
    exact_energy = float(agp_energy(pair_ground, pairs, one_body_cpu).detach())
    finite_truth = float(
        torch.linalg.eigvalsh(
            antisymmetric_many_body_hamiltonian(one_body_cpu, particles)
        )[0]
        .real.detach()
    )

    raw_ground = (0.5 * pair_ground).requires_grad_(True)
    ground_stationary_energy = agp_energy(
        raw_ground - raw_ground.transpose(0, 1), pairs, one_body_cpu
    )
    ground_gradient = torch.autograd.grad(ground_stationary_energy, raw_ground)[0]

    spectra = []
    for cut in range(1, particles):
        observed = particle_schmidt_spectrum(slater, cut)
        expected = slater_flat_spectrum(particles, cut, dtype=observed.dtype)
        nonzero = observed[: expected.numel()]
        spectra.append(
            {
                "cut": cut,
                "multiplicity": expected.numel(),
                "expected_singular_value": float(expected[0]),
                "max_abs_error": float(torch.max(torch.abs(nonzero - expected))),
            }
        )
    central_spectrum = particle_schmidt_spectrum(slater, 2)
    central_truncation = [
        {
            "rank": rank,
            "observed_relative_error": float(
                best_rank_error(central_spectrum, rank)
            ),
            "closed_form_relative_error": math.sqrt((6 - rank) / 6),
        }
        for rank in range(1, 6)
    ]

    raw = torch.nn.Parameter(_random_raw(config.basis_order, config.seed, device))
    optimizer = torch.optim.Adam([raw], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.final_learning_rate,
    )

    def energy() -> torch.Tensor:
        pair_matrix = raw - raw.transpose(0, 1)
        return agp_energy(pair_matrix, pairs, one_body)

    interval = max(1, config.steps // config.record_points)
    history = [
        {
            "step": 0,
            "energy": float(energy().detach().cpu()),
            "learning_rate": config.learning_rate,
        }
    ]
    best_energy = history[0]["energy"]
    best_raw = raw.detach().clone()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        loss = energy()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            raw.div_(torch.linalg.vector_norm(raw - raw.transpose(0, 1)))
        scheduler.step()
        if step % interval == 0 or step == config.steps:
            observed_energy = float(energy().detach().cpu())
            history.append(
                {
                    "step": step,
                    "energy": observed_energy,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )
            if observed_energy < best_energy:
                best_energy = observed_energy
                best_raw = raw.detach().clone()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    terminal_energy = float(energy().detach().cpu())
    with torch.no_grad():
        raw.copy_(best_raw)
    final_energy = float(energy().detach().cpu())
    optimized_state = agp_tensor(
        (raw - raw.transpose(0, 1)).detach().cpu(), pairs
    )
    fidelity = float(
        (
            torch.abs(torch.vdot(slater.reshape(-1), optimized_state.reshape(-1)))
            ** 2
            / torch.vdot(optimized_state.reshape(-1), optimized_state.reshape(-1)).real
        ).detach()
    )

    record = {
        "schema_version": 1,
        "experiment": "functional_pfaffian_e3_four_noninteracting_fermions",
        "config": asdict(config),
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "compute_capability": (
                torch.cuda.get_device_capability(device)
                if device.type == "cuda"
                else None
            ),
        },
        "continuum_reference_energy": reference_energy,
        "finite_basis_reference_energy": finite_truth,
        "constructed_agp_energy": exact_energy,
        "constructed_energy_absolute_error": abs(exact_energy - reference_energy),
        "chi_one_femps_max_abs_error": float(torch.max(torch.abs(slater_femps - slater))),
        "pair_channel_femps_max_abs_error": float(torch.max(torch.abs(agp_femps - slater))),
        "pfaffian_tensor_max_abs_error": float(torch.max(torch.abs(agp_ground - slater))),
        "ground_energy_gradient_norm": float(torch.linalg.vector_norm(ground_gradient)),
        "antisymmetry_residual": float(antisymmetry_residual(slater)),
        "ordinary_particle_tt_ranks": list(particle_tt_ranks(slater)),
        "femps_correlation_bonds": [1, 1, 1],
        "pfaffian_pair_channels": 2,
        "schmidt_spectra": spectra,
        "central_cut_truncation": central_truncation,
        "blind_ad": {
            "initial_energy": history[0]["energy"],
            "terminal_energy_before_best_restore": terminal_energy,
            "final_energy": final_energy,
            "error_vs_reference": abs(final_energy - reference_energy),
            "ground_state_fidelity": fidelity,
            "history": history,
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    passed = (
        record["constructed_energy_absolute_error"] < 1e-12
        and record["ordinary_particle_tt_ranks"] == [4, 6, 4]
        and record["blind_ad"]["error_vs_reference"] < 1e-9
        and record["blind_ad"]["ground_state_fidelity"] > 1 - 1e-9
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
