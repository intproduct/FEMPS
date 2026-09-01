"""Controlled coordinate-space VMC for explicit-correlator exterior carriers.

The state remains first quantized and continuous:

``Psi(X) = exp(J_theta(X)) det[psi_a(x_i)] / sqrt(N!)``.

The determinant is the ``chi=1`` exterior carrier; the symmetric pair factor
is an independent continuous correlation control.  This module does not form a
``D**N`` coefficient tensor or convert the state to occupation-number MPS.
The initial backend is deliberately CPU/float64 and aimed at fixed-state
estimator validation before any interacting ``N=4`` production run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from .correlated_exterior import harmonic_function_values_and_derivatives


VMC_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CorrelatedExteriorVMCConfig:
    """Deterministic multi-chain random-walk Metropolis configuration."""

    particles: int
    chains: int
    burn_in_sweeps: int
    samples_per_chain: int
    thinning_sweeps: int
    proposal_scale: float
    seed: int
    max_autocorrelation_lag: int = 100
    checkpoint_every: int = 100
    omega: float = 1.0
    coupling: float = 1.0
    softening: float = 1.0
    device: str = "cpu"

    def validate(self) -> None:
        if min(
            self.particles,
            self.chains,
            self.burn_in_sweeps,
            self.samples_per_chain,
            self.thinning_sweeps,
            self.max_autocorrelation_lag,
            self.checkpoint_every,
        ) < 1:
            raise ValueError("particle, chain, sweep, sample, and lag counts must be positive")
        if not math.isfinite(self.proposal_scale) or self.proposal_scale <= 0:
            raise ValueError("proposal_scale must be finite and positive")
        if not math.isfinite(self.omega) or self.omega <= 0:
            raise ValueError("omega must be finite and positive")
        if not math.isfinite(self.coupling) or self.coupling < 0:
            raise ValueError("coupling must be finite and nonnegative")
        if not math.isfinite(self.softening) or self.softening <= 0:
            raise ValueError("softening must be finite and positive")
        if self.device != "cpu":
            raise ValueError("the Phase 43 validation backend is frozen to CPU")


@dataclass(frozen=True, slots=True)
class LocalEnergyResult:
    """Pointwise wavefunction and Hamiltonian diagnostics."""

    local_energy: torch.Tensor
    kinetic_energy: torch.Tensor
    trap_energy: torch.Tensor
    interaction_energy: torch.Tensor
    log_abs_wavefunction: torch.Tensor
    sign: torch.Tensor


def canonical_exterior_carrier(raw: torch.Tensor) -> torch.Tensor:
    """Return a deterministic real Stiefel gauge for a ``(D,N)`` carrier."""

    if raw.ndim != 2 or raw.shape[0] < raw.shape[1] or raw.shape[1] < 1:
        raise ValueError("raw carrier must have shape (D,N) with D >= N >= 1")
    if raw.dtype != torch.float64:
        raise TypeError("the Phase 43 carrier requires float64")
    q, r = torch.linalg.qr(raw, mode="reduced")
    diagonal = torch.diagonal(r)
    if bool(torch.any(torch.abs(diagonal) <= torch.finfo(raw.dtype).tiny)):
        raise ValueError("raw carrier is rank deficient")
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return (q * signs[None, :]).contiguous()


def canonical_lowest_orbitals(
    basis_order: int, particles: int, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Return the lowest harmonic orbitals as a functional-basis carrier."""

    if basis_order < particles or particles < 1:
        raise ValueError("require D >= N >= 1")
    if dtype != torch.float64:
        raise TypeError("the Phase 43 carrier requires float64")
    orbitals = torch.zeros((basis_order, particles), dtype=dtype)
    orbitals[:particles] = torch.eye(particles, dtype=dtype)
    return orbitals


def _validate_state(
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> None:
    if orbitals.ndim != 2 or orbitals.shape[0] < orbitals.shape[1]:
        raise ValueError("orbitals must have shape (D,N) with D >= N")
    if orbitals.dtype != torch.float64:
        raise TypeError("orbitals must use float64")
    if amplitudes.ndim != 1 or exponents.ndim != 1 or amplitudes.shape != exponents.shape:
        raise ValueError("correlator amplitudes/exponents must be equal vectors")
    if amplitudes.dtype != torch.float64 or exponents.dtype != torch.float64:
        raise TypeError("correlator tensors must use float64")
    if orbitals.device != amplitudes.device or orbitals.device != exponents.device:
        raise ValueError("carrier and correlator tensors must share a device")
    if exponents.numel() and bool(torch.any(exponents <= 0)):
        raise ValueError("Gaussian pair exponents must be positive")
    if positions is not None:
        if positions.ndim < 1 or positions.shape[-1] != orbitals.shape[1]:
            raise ValueError("positions must have final dimension N")
        if positions.dtype != torch.float64 or positions.device != orbitals.device:
            raise TypeError("positions must match the state float64 device")


def _functional_orbital_tables(
    orbitals: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return orbital values and first/second coordinate derivatives."""

    flat = positions.reshape(-1)
    basis, basis_derivative = harmonic_function_values_and_derivatives(
        orbitals.shape[0], flat
    )
    levels = torch.arange(
        orbitals.shape[0], dtype=torch.float64, device=positions.device
    )
    basis_second = (
        flat[None, :].square() - (2.0 * levels[:, None] + 1.0)
    ) * basis
    leading = positions.shape[:-1]
    particles = positions.shape[-1]
    basis = basis.reshape((orbitals.shape[0], *leading, particles))
    basis_derivative = basis_derivative.reshape(
        (orbitals.shape[0], *leading, particles)
    )
    basis_second = basis_second.reshape((orbitals.shape[0], *leading, particles))
    values = torch.einsum("da,d...i->...ia", orbitals, basis)
    derivatives = torch.einsum("da,d...i->...ia", orbitals, basis_derivative)
    second_derivatives = torch.einsum("da,d...i->...ia", orbitals, basis_second)
    return values, derivatives, second_derivatives


def gaussian_pair_log_correlator(
    positions: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``J``, its coordinate gradient, and diagonal Laplacian terms."""

    if amplitudes.shape != exponents.shape or amplitudes.ndim != 1:
        raise ValueError("correlator amplitudes/exponents must be equal vectors")
    particles = positions.shape[-1]
    separation = positions[..., :, None] - positions[..., None, :]
    mask = 1.0 - torch.eye(
        particles, dtype=torch.float64, device=positions.device
    )
    if amplitudes.numel() == 0:
        log_value = torch.zeros(positions.shape[:-1], dtype=torch.float64, device=positions.device)
        zeros = torch.zeros_like(positions)
        return log_value, zeros, zeros
    gaussian = torch.exp(
        -exponents[:, None, None] * separation.square()[..., None, :, :]
    )
    weighted = amplitudes[:, None, None] * gaussian
    log_value = 0.5 * torch.sum(weighted * mask, dim=(-3, -2, -1))
    first_kernel = (
        -2.0
        * exponents[:, None, None]
        * separation[..., None, :, :]
        * gaussian
    )
    second_kernel = (
        -2.0 * exponents[:, None, None]
        + 4.0
        * exponents[:, None, None].square()
        * separation[..., None, :, :].square()
    ) * gaussian
    gradient = torch.sum(
        amplitudes[:, None, None] * first_kernel * mask, dim=(-3, -1)
    )
    laplacian = torch.sum(
        amplitudes[:, None, None] * second_kernel * mask, dim=(-3, -1)
    )
    return log_value, gradient, laplacian


def log_abs_correlated_exterior_wavefunction(
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized log magnitude and determinant sign at coordinates."""

    _validate_state(orbitals, amplitudes, exponents, positions)
    values = _functional_orbital_tables(orbitals, positions)[0]
    sign, log_abs_determinant = torch.linalg.slogdet(values)
    log_jastrow = gaussian_pair_log_correlator(
        positions, amplitudes, exponents
    )[0]
    normalization = 0.5 * math.lgamma(orbitals.shape[1] + 1.0)
    return log_jastrow + log_abs_determinant - normalization, sign


def correlated_exterior_wavefunction_value(
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Return the signed continuous first-quantized wavefunction value."""

    log_abs, sign = log_abs_correlated_exterior_wavefunction(
        orbitals, amplitudes, exponents, positions
    )
    return sign * torch.exp(log_abs)


def correlated_exterior_local_energy(
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    positions: torch.Tensor,
    *,
    omega: float = 1.0,
    coupling: float = 1.0,
    softening: float = 1.0,
) -> LocalEnergyResult:
    """Evaluate the analytic local energy away from determinant nodes."""

    _validate_state(orbitals, amplitudes, exponents, positions)
    if not math.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and positive")
    if not math.isfinite(coupling) or coupling < 0:
        raise ValueError("coupling must be finite and nonnegative")
    if not math.isfinite(softening) or softening <= 0:
        raise ValueError("softening must be finite and positive")
    values, derivatives, second_derivatives = _functional_orbital_tables(
        orbitals, positions
    )
    sign, log_abs_determinant = torch.linalg.slogdet(values)
    # Row scaling cancels exactly in all determinant derivative ratios and
    # prevents false singular pivots when harmonic functions are tiny in the
    # quadrature/sampling tails.
    row_scale = torch.amax(torch.abs(values), dim=-1).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    scaled_values = values / row_scale[..., :, None]
    scaled_derivatives = derivatives / row_scale[..., :, None]
    scaled_second_derivatives = second_derivatives / row_scale[..., :, None]
    inverse = torch.linalg.inv(scaled_values)
    slater_gradient = torch.einsum(
        "...ai,...ia->...i", inverse, scaled_derivatives
    )
    slater_second_ratio = torch.einsum(
        "...ai,...ia->...i", inverse, scaled_second_derivatives
    )
    slater_log_laplacian = slater_second_ratio - slater_gradient.square()
    log_jastrow, jastrow_gradient, jastrow_laplacian = gaussian_pair_log_correlator(
        positions, amplitudes, exponents
    )
    total_gradient = slater_gradient + jastrow_gradient
    total_log_laplacian = slater_log_laplacian + jastrow_laplacian
    kinetic = -0.5 * torch.sum(
        total_log_laplacian + total_gradient.square(), dim=-1
    )
    trap = 0.5 * omega * omega * torch.sum(positions.square(), dim=-1)
    separation = positions[..., :, None] - positions[..., None, :]
    particles = positions.shape[-1]
    upper = torch.triu(
        torch.ones(
            (particles, particles), dtype=torch.bool, device=positions.device
        ),
        diagonal=1,
    )
    pair_potential = torch.rsqrt(separation.square() + softening * softening)
    interaction = coupling * torch.sum(pair_potential[..., upper], dim=-1)
    normalization = 0.5 * math.lgamma(orbitals.shape[1] + 1.0)
    log_abs = log_jastrow + log_abs_determinant - normalization
    return LocalEnergyResult(
        local_energy=kinetic + trap + interaction,
        kinetic_energy=kinetic,
        trap_energy=trap,
        interaction_energy=interaction,
        log_abs_wavefunction=log_abs,
        sign=sign,
    )


def sampled_antisymmetry_residual(
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, float]:
    """Measure swap antisymmetry and correlator symmetry on sampled states."""

    if positions.shape[-1] < 2:
        return {"antisymmetry_residual": 0.0, "correlator_symmetry_residual": 0.0}
    swapped = positions.clone()
    swapped[..., 0] = positions[..., 1]
    swapped[..., 1] = positions[..., 0]
    value = correlated_exterior_wavefunction_value(
        orbitals, amplitudes, exponents, positions
    )
    swapped_value = correlated_exterior_wavefunction_value(
        orbitals, amplitudes, exponents, swapped
    )
    scale = torch.max(torch.abs(value)).clamp_min(torch.finfo(torch.float64).tiny)
    antisymmetry = torch.max(torch.abs(value + swapped_value)) / scale
    log_jastrow = gaussian_pair_log_correlator(positions, amplitudes, exponents)[0]
    swapped_log_jastrow = gaussian_pair_log_correlator(
        swapped, amplitudes, exponents
    )[0]
    correlator_scale = torch.max(torch.abs(log_jastrow)).clamp_min(1.0)
    correlator_symmetry = torch.max(
        torch.abs(log_jastrow - swapped_log_jastrow)
    ) / correlator_scale
    return {
        "antisymmetry_residual": float(antisymmetry),
        "correlator_symmetry_residual": float(correlator_symmetry),
    }


def _state_sha256(
    orbitals: torch.Tensor, amplitudes: torch.Tensor, exponents: torch.Tensor
) -> str:
    digest = hashlib.sha256()
    for tensor in (orbitals, amplitudes, exponents):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _log_probability(
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    return 2.0 * log_abs_correlated_exterior_wavefunction(
        orbitals, amplitudes, exponents, positions
    )[0]


def _metropolis_sweep(
    positions: torch.Tensor,
    log_probability: torch.Tensor,
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    proposal_scale: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    accepted = 0
    proposed = 0
    for particle in range(positions.shape[-1]):
        candidate = positions.clone()
        candidate[:, particle] = candidate[:, particle] + proposal_scale * torch.randn(
            positions.shape[0], generator=generator, dtype=torch.float64
        )
        candidate_log_probability = _log_probability(
            orbitals, amplitudes, exponents, candidate
        )
        log_uniform = torch.log(
            torch.rand(
                positions.shape[0], generator=generator, dtype=torch.float64
            ).clamp_min(torch.finfo(torch.float64).tiny)
        )
        accept = log_uniform < (candidate_log_probability - log_probability)
        positions = torch.where(accept[:, None], candidate, positions)
        log_probability = torch.where(
            accept, candidate_log_probability, log_probability
        )
        accepted += int(torch.count_nonzero(accept))
        proposed += positions.shape[0]
    return positions, log_probability, accepted, proposed


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_correlated_exterior_vmc_checkpoint(path: Path) -> dict[str, Any]:
    """Load a Phase 43 VMC checkpoint without accepting incompatible data."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("VMC checkpoint payload must be a mapping")
    if payload.get("schema_version") != VMC_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported VMC checkpoint schema")
    if payload.get("method") != "correlated_exterior_coordinate_vmc":
        raise ValueError("unexpected VMC checkpoint method")
    return payload


def run_correlated_exterior_vmc(
    config: CorrelatedExteriorVMCConfig,
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    max_samples_this_call: int | None = None,
) -> dict[str, Any]:
    """Sample a fixed state with exact checkpoint/resume RNG continuation."""

    config.validate()
    _validate_state(orbitals, amplitudes, exponents)
    if orbitals.shape[1] != config.particles:
        raise ValueError("VMC config particle count does not match the carrier")
    if max_samples_this_call is not None and max_samples_this_call < 1:
        raise ValueError("max_samples_this_call must be positive")
    state_hash = _state_sha256(orbitals, amplitudes, exponents)
    generator = torch.Generator().manual_seed(config.seed)
    if resume:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise ValueError("resume requires an existing checkpoint")
        payload = load_correlated_exterior_vmc_checkpoint(checkpoint_path)
        if payload["config"] != asdict(config):
            raise ValueError("VMC checkpoint config mismatch")
        if payload["state_sha256"] != state_hash:
            raise ValueError("VMC checkpoint state mismatch")
        positions = payload["positions"].clone()
        log_probability = payload["log_probability"].clone()
        samples = payload["samples"].clone()
        accepted = int(payload["accepted_proposals"])
        proposed = int(payload["total_proposals"])
        burn_in_completed = bool(payload["burn_in_completed"])
        generator.set_state(payload["generator_state"])
    else:
        positions = torch.randn(
            (config.chains, config.particles),
            generator=generator,
            dtype=torch.float64,
        )
        log_probability = _log_probability(
            orbitals, amplitudes, exponents, positions
        )
        samples = torch.empty(
            (0, config.chains, config.particles), dtype=torch.float64
        )
        accepted = 0
        proposed = 0
        burn_in_completed = False
    if not burn_in_completed:
        for _ in range(config.burn_in_sweeps):
            positions, log_probability, count, attempts = _metropolis_sweep(
                positions,
                log_probability,
                orbitals,
                amplitudes,
                exponents,
                config.proposal_scale,
                generator,
            )
            accepted += count
            proposed += attempts
        burn_in_completed = True

    remaining = config.samples_per_chain - samples.shape[0]
    admitted = remaining
    if max_samples_this_call is not None:
        admitted = min(admitted, max_samples_this_call)
    new_samples = []
    for index in range(admitted):
        for _ in range(config.thinning_sweeps):
            positions, log_probability, count, attempts = _metropolis_sweep(
                positions,
                log_probability,
                orbitals,
                amplitudes,
                exponents,
                config.proposal_scale,
                generator,
            )
            accepted += count
            proposed += attempts
        new_samples.append(positions.clone())
        completed_count = samples.shape[0] + index + 1
        if checkpoint_path is not None and completed_count % config.checkpoint_every == 0:
            checkpoint_samples = torch.cat(
                (samples, torch.stack(new_samples)), dim=0
            )
            _save_checkpoint(
                checkpoint_path,
                {
                    "schema_version": VMC_CHECKPOINT_SCHEMA_VERSION,
                    "method": "correlated_exterior_coordinate_vmc",
                    "config": asdict(config),
                    "state_sha256": state_hash,
                    "positions": positions,
                    "log_probability": log_probability,
                    "samples": checkpoint_samples,
                    "accepted_proposals": accepted,
                    "total_proposals": proposed,
                    "burn_in_completed": burn_in_completed,
                    "generator_state": generator.get_state(),
                },
            )
    if new_samples:
        samples = torch.cat((samples, torch.stack(new_samples)), dim=0)
    completed = samples.shape[0] == config.samples_per_chain
    payload = {
        "schema_version": VMC_CHECKPOINT_SCHEMA_VERSION,
        "method": "correlated_exterior_coordinate_vmc",
        "config": asdict(config),
        "state_sha256": state_hash,
        "positions": positions,
        "log_probability": log_probability,
        "samples": samples,
        "accepted_proposals": accepted,
        "total_proposals": proposed,
        "burn_in_completed": burn_in_completed,
        "generator_state": generator.get_state(),
    }
    if checkpoint_path is not None:
        _save_checkpoint(checkpoint_path, payload)
    result = vmc_observables(config, orbitals, amplitudes, exponents, samples)
    result.update(
        {
            "completed": completed,
            "completed_samples_per_chain": int(samples.shape[0]),
            "acceptance_rate": accepted / proposed if proposed else 0.0,
            "accepted_proposals": accepted,
            "total_proposals": proposed,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "samples": samples,
        }
    )
    return result


def _integrated_autocorrelation_time(
    values: torch.Tensor, maximum_lag: int
) -> float:
    centered = values - torch.mean(values)
    variance = torch.mean(centered.square())
    if float(variance) <= 100 * torch.finfo(torch.float64).eps:
        return 1.0
    tau = 1.0
    maximum = min(maximum_lag, values.numel() - 1)
    for lag in range(1, maximum + 1):
        correlation = torch.mean(centered[:-lag] * centered[lag:]) / variance
        value = float(correlation)
        if value <= 0:
            break
        tau += 2.0 * value
    return max(1.0, tau)


def _split_rhat(values: torch.Tensor) -> float:
    """Return a chain-mean Gelman--Rubin diagnostic for ``(S,C)`` data."""

    samples, chains = values.shape
    if samples < 2 or chains < 2:
        return 1.0
    chain_means = torch.mean(values, dim=0)
    within = torch.mean(torch.var(values, dim=0, unbiased=True))
    between = samples * torch.var(chain_means, unbiased=True)
    tiny = 100 * torch.finfo(torch.float64).eps
    if float(within) <= tiny:
        return 1.0 if float(between) <= tiny else math.inf
    variance = ((samples - 1.0) / samples) * within + between / samples
    return float(torch.sqrt(variance / within))


def _blocking_standard_error(
    values: torch.Tensor, block_size: int
) -> tuple[float, int]:
    samples, chains = values.shape
    blocks_per_chain = samples // block_size
    if blocks_per_chain < 2:
        return float(torch.std(torch.mean(values, dim=0), unbiased=True) / math.sqrt(chains)) if chains > 1 else 0.0, blocks_per_chain
    truncated = values[: blocks_per_chain * block_size]
    block_means = truncated.reshape(blocks_per_chain, block_size, chains).mean(dim=1)
    flattened = block_means.reshape(-1)
    error = torch.std(flattened, unbiased=True) / math.sqrt(flattened.numel())
    return float(error), blocks_per_chain


def vmc_observables(
    config: CorrelatedExteriorVMCConfig,
    orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    samples: torch.Tensor,
) -> dict[str, Any]:
    """Compute energy, uncertainty, mixing, and symmetry diagnostics."""

    if samples.ndim != 3 or samples.shape[1:] != (
        config.chains,
        config.particles,
    ):
        raise ValueError("samples must have shape (S,chains,N)")
    if samples.shape[0] < 1:
        return {
            "energy": None,
            "energy_variance": None,
            "energy_standard_error": None,
            "blocking_standard_error": None,
            "effective_sample_size": 0.0,
            "integrated_autocorrelation_times": [],
            "rhat": None,
            "chain_means": [],
            "symmetry": None,
        }
    local = correlated_exterior_local_energy(
        orbitals,
        amplitudes,
        exponents,
        samples,
        omega=config.omega,
        coupling=config.coupling,
        softening=config.softening,
    )
    energies = local.local_energy
    energy = torch.mean(energies)
    variance = torch.var(energies, unbiased=True) if energies.numel() > 1 else energies.new_zeros(())
    taus = [
        _integrated_autocorrelation_time(
            energies[:, chain], config.max_autocorrelation_lag
        )
        for chain in range(config.chains)
    ]
    effective_sample_size = sum(samples.shape[0] / tau for tau in taus)
    ess_error = math.sqrt(float(variance) / effective_sample_size) if effective_sample_size else math.inf
    block_size = max(1, int(math.ceil(2.0 * max(taus))))
    blocking_error, blocks_per_chain = _blocking_standard_error(energies, block_size)
    chain_means = torch.mean(energies, dim=0)
    chain_error = (
        float(torch.std(chain_means, unbiased=True) / math.sqrt(config.chains))
        if config.chains > 1
        else 0.0
    )
    symmetry = sampled_antisymmetry_residual(
        orbitals, amplitudes, exponents, samples
    )
    return {
        "energy": float(energy),
        "kinetic_energy": float(torch.mean(local.kinetic_energy)),
        "trap_energy": float(torch.mean(local.trap_energy)),
        "interaction_energy": float(torch.mean(local.interaction_energy)),
        "energy_variance": float(variance),
        "energy_standard_error": max(ess_error, blocking_error, chain_error),
        "ess_standard_error": ess_error,
        "blocking_standard_error": blocking_error,
        "chain_mean_standard_error": chain_error,
        "blocking_size": block_size,
        "blocks_per_chain": blocks_per_chain,
        "effective_sample_size": effective_sample_size,
        "integrated_autocorrelation_times": taus,
        "rhat": _split_rhat(energies),
        "chain_means": chain_means.tolist(),
        "symmetry": symmetry,
    }


def vmc_energy_gradient(
    config: CorrelatedExteriorVMCConfig,
    raw_orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    samples: torch.Tensor,
) -> dict[str, Any]:
    """Return covariance/log-derivative energy gradients and chain errors."""

    raw = raw_orbitals.detach().clone().contiguous().requires_grad_(True)
    amp = amplitudes.detach().clone().contiguous().requires_grad_(True)
    orbitals = canonical_exterior_carrier(raw)
    with torch.no_grad():
        local_energy = correlated_exterior_local_energy(
            orbitals.detach(),
            amp.detach(),
            exponents,
            samples,
            omega=config.omega,
            coupling=config.coupling,
            softening=config.softening,
        ).local_energy
    log_abs = log_abs_correlated_exterior_wavefunction(
        orbitals, amp, exponents, samples
    )[0]
    centered = (local_energy - torch.mean(local_energy)).detach()
    loss = 2.0 * torch.mean(centered * log_abs)
    orbital_gradient, amplitude_gradient = torch.autograd.grad(
        loss, (raw, amp), retain_graph=True, allow_unused=True
    )
    if orbital_gradient is None:
        orbital_gradient = torch.zeros_like(raw)
    if amplitude_gradient is None:
        amplitude_gradient = torch.zeros_like(amp)
    orbital_chain = []
    amplitude_chain = []
    for chain in range(config.chains):
        chain_energy = local_energy[:, chain]
        chain_centered = (chain_energy - torch.mean(chain_energy)).detach()
        chain_loss = 2.0 * torch.mean(chain_centered * log_abs[:, chain])
        gradients = torch.autograd.grad(
            chain_loss, (raw, amp), retain_graph=True, allow_unused=True
        )
        orbital_chain.append(
            torch.zeros_like(raw) if gradients[0] is None else gradients[0]
        )
        amplitude_chain.append(
            torch.zeros_like(amp) if gradients[1] is None else gradients[1]
        )
    orbital_chain_tensor = torch.stack(orbital_chain)
    amplitude_chain_tensor = torch.stack(amplitude_chain)
    orbital_standard_error = (
        torch.std(orbital_chain_tensor, dim=0, unbiased=True)
        / math.sqrt(config.chains)
        if config.chains > 1
        else torch.zeros_like(orbital_gradient)
    )
    amplitude_standard_error = (
        torch.std(amplitude_chain_tensor, dim=0, unbiased=True)
        / math.sqrt(config.chains)
        if config.chains > 1
        else torch.zeros_like(amplitude_gradient)
    )
    return {
        "orbital_gradient": orbital_gradient.detach(),
        "amplitude_gradient": amplitude_gradient.detach(),
        "orbital_chain_standard_error": orbital_standard_error.detach(),
        "amplitude_chain_standard_error": amplitude_standard_error.detach(),
        "chain_orbital_gradients": orbital_chain_tensor.detach(),
        "chain_amplitude_gradients": amplitude_chain_tensor.detach(),
    }


__all__ = [
    "VMC_CHECKPOINT_SCHEMA_VERSION",
    "CorrelatedExteriorVMCConfig",
    "LocalEnergyResult",
    "canonical_exterior_carrier",
    "canonical_lowest_orbitals",
    "correlated_exterior_local_energy",
    "correlated_exterior_wavefunction_value",
    "gaussian_pair_log_correlator",
    "load_correlated_exterior_vmc_checkpoint",
    "log_abs_correlated_exterior_wavefunction",
    "run_correlated_exterior_vmc",
    "sampled_antisymmetry_residual",
    "vmc_energy_gradient",
    "vmc_observables",
]
