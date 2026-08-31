"""Opt-in CPU/Blackwell parity check for the functional-MPS energy and gradient."""

from __future__ import annotations

import json

import torch

from femps.baselines.coupled_oscillators import functional_mps_energy
from femps.baselines.functional_mps import random_functional_mps
from femps.devices import resolve_device
from latticetn.mps import MPS


def main() -> int:
    gpu = resolve_device("auto")
    if gpu.type != "cuda":
        raise RuntimeError("no CUDA device is available")

    cpu_mps = random_functional_mps(3, 4, 6, seed=17, device="cpu")
    gpu_mps = MPS.from_tensors(
        [tensor.to(gpu) for tensor in cpu_mps.tensors],
        dtype=torch.complex128,
        device=gpu,
        requires_grad=True,
    )

    cpu_energy = functional_mps_energy(cpu_mps, gamma=-0.2)
    gpu_energy = functional_mps_energy(gpu_mps, gamma=-0.2)
    cpu_energy.backward()
    gpu_energy.backward()

    cpu_energy_value = float(cpu_energy.detach())
    gpu_energy_value = float(gpu_energy.detach().cpu())

    gradient_max_abs_diff = max(
        float((cpu.grad - gpu_tensor.grad.cpu()).abs().max())
        for cpu, gpu_tensor in zip(cpu_mps.tensors, gpu_mps.tensors)
    )
    result = {
        "device": str(gpu),
        "gpu": torch.cuda.get_device_name(gpu),
        "compute_capability": torch.cuda.get_device_capability(gpu),
        "cpu_energy": cpu_energy_value,
        "gpu_energy": gpu_energy_value,
        "energy_abs_diff": abs(cpu_energy_value - gpu_energy_value),
        "gradient_max_abs_diff": gradient_max_abs_diff,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["energy_abs_diff"] > 1e-11 or gradient_max_abs_diff > 1e-10:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
