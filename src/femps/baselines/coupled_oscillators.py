"""Hong et al. coupled-oscillator functional-MPS baseline."""

from __future__ import annotations

import math

import torch

from femps.basis.harmonic import harmonic_hamiltonian, position_matrix


def normal_mode_frequencies(
    num_oscillators: int,
    *,
    gamma: float,
    omega: float = 1.0,
) -> torch.Tensor:
    """Exact open-chain mode frequencies for nearest-neighbor ``gamma*x*x``."""
    if num_oscillators < 1:
        raise ValueError("num_oscillators must be positive")
    n = torch.arange(1, num_oscillators + 1, dtype=torch.float64)
    squared = omega * omega + 2.0 * gamma * torch.cos(
        n * math.pi / (num_oscillators + 1)
    )
    if torch.any(squared <= 0):
        raise ValueError("the quadratic Hamiltonian is not positive definite")
    return torch.sqrt(squared)


def exact_ground_energy(
    num_oscillators: int,
    *,
    gamma: float,
    omega: float = 1.0,
) -> float:
    """Continuum normal-mode ground energy, Eq. (31) of arXiv:2201.12823."""
    return float(0.5 * normal_mode_frequencies(
        num_oscillators, gamma=gamma, omega=omega
    ).sum())


def functional_mps_energy(
    mps,
    *,
    gamma: float,
    omega: float = 1.0,
) -> torch.Tensor:
    """Differentiable Rayleigh energy using latticeTN native contractions.

    This implements Eq. (27) with nearest-neighbor two-body coupling and
    ``tilde_gamma=0``. Expectations are accumulated as raw numerators and
    divided by the native MPS norm exactly once.
    """
    try:
        from latticetn.contractions import (
            native_local_expect,
            native_norm_sq,
            native_two_site_expect,
        )
    except ImportError as exc:
        raise ImportError("functional_mps_energy requires the sibling latticeTN package") from exc

    tensor0 = mps.tensors[0]
    h1 = harmonic_hamiltonian(
        mps.dim, omega=omega, dtype=tensor0.dtype, device=tensor0.device
    )
    x = position_matrix(
        mps.dim, dtype=tensor0.dtype, device=tensor0.device
    )
    numerator = torch.zeros((), dtype=tensor0.dtype, device=tensor0.device)
    for site in range(mps.N):
        numerator = numerator + native_local_expect(mps, h1, site)
    for site in range(mps.N - 1):
        numerator = numerator + gamma * native_two_site_expect(
            mps, x, site, x, site + 1
        )
    return (numerator / native_norm_sq(mps)).real


def dense_truncated_hamiltonian(
    num_oscillators: int,
    basis_order: int,
    *,
    gamma: float,
    omega: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Small-system truth oracle; scales as ``basis_order**num_oscillators``."""
    if num_oscillators < 1:
        raise ValueError("num_oscillators must be positive")
    identity = torch.eye(basis_order, dtype=dtype, device=device)
    h1 = harmonic_hamiltonian(basis_order, omega=omega, dtype=dtype, device=device)
    x = position_matrix(basis_order, dtype=dtype, device=device)

    def kron_sites(operators: dict[int, torch.Tensor]) -> torch.Tensor:
        result = torch.ones((1, 1), dtype=dtype, device=device)
        for site in range(num_oscillators):
            result = torch.kron(result, operators.get(site, identity))
        return result

    dimension = basis_order ** num_oscillators
    hamiltonian = torch.zeros((dimension, dimension), dtype=dtype, device=device)
    for site in range(num_oscillators):
        hamiltonian = hamiltonian + kron_sites({site: h1})
    for site in range(num_oscillators - 1):
        hamiltonian = hamiltonian + gamma * kron_sites({site: x, site + 1: x})
    return hamiltonian

