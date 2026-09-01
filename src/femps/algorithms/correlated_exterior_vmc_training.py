"""Checkpointed stochastic optimization for correlated exterior VMC states.

This module builds on the fixed-state Phase 43 estimator without modifying its
source-hashed validation implementation. It performs ambient Adam updates of a
real functional-orbital carrier followed by a deterministic QR retraction, and
updates symmetric pair-correlator amplitudes directly. The target remains a
continuous first-quantized wavefunction throughout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from .correlated_exterior_vmc import (
    CorrelatedExteriorVMCConfig,
    _log_probability,
    _metropolis_sweep,
    _save_checkpoint,
    canonical_exterior_carrier,
    vmc_energy_gradient,
    vmc_observables,
)


VMC_OPTIMIZER_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CorrelatedExteriorVMCOptimizerConfig:
    """Frozen online-sampling Adam/retraction optimization controls."""

    particles: int
    chains: int
    steps: int
    burn_in_sweeps: int
    rethermalization_sweeps: int
    samples_per_chain: int
    thinning_sweeps: int
    proposal_scale: float
    seed: int
    learning_rate: float
    final_learning_rate: float
    gradient_clip_norm: float
    amplitude_bound: float
    checkpoint_every: int = 10
    max_autocorrelation_lag: int = 50
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    omega: float = 1.0
    coupling: float = 1.0
    softening: float = 1.0
    device: str = "cpu"

    def validate(self) -> None:
        integer_controls = (
            self.particles,
            self.chains,
            self.steps,
            self.burn_in_sweeps,
            self.rethermalization_sweeps,
            self.samples_per_chain,
            self.thinning_sweeps,
            self.checkpoint_every,
            self.max_autocorrelation_lag,
        )
        if min(integer_controls) < 1:
            raise ValueError("all optimizer counts must be positive")
        positive_controls = (
            self.proposal_scale,
            self.learning_rate,
            self.final_learning_rate,
            self.gradient_clip_norm,
            self.amplitude_bound,
            self.adam_epsilon,
            self.omega,
            self.softening,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive_controls):
            raise ValueError("optimizer scales must be finite and positive")
        if not (0 <= self.adam_beta1 < 1 and 0 <= self.adam_beta2 < 1):
            raise ValueError("Adam beta values must lie in [0,1)")
        if not math.isfinite(self.coupling) or self.coupling < 0:
            raise ValueError("coupling must be finite and nonnegative")
        if self.device != "cpu":
            raise ValueError("the initial stochastic optimizer is frozen to CPU")

    def estimator_config(self) -> CorrelatedExteriorVMCConfig:
        """Return the per-step estimator configuration."""

        return CorrelatedExteriorVMCConfig(
            particles=self.particles,
            chains=self.chains,
            burn_in_sweeps=1,
            samples_per_chain=self.samples_per_chain,
            thinning_sweeps=self.thinning_sweeps,
            proposal_scale=self.proposal_scale,
            seed=self.seed,
            max_autocorrelation_lag=self.max_autocorrelation_lag,
            checkpoint_every=self.samples_per_chain,
            omega=self.omega,
            coupling=self.coupling,
            softening=self.softening,
            device=self.device,
        )


def _tensor_state_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _learning_rate(config: CorrelatedExteriorVMCOptimizerConfig, step: int) -> float:
    if config.steps == 1:
        return config.final_learning_rate
    fraction = step / (config.steps - 1)
    return config.learning_rate * (
        config.final_learning_rate / config.learning_rate
    ) ** fraction


def _load_optimizer_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("VMC optimizer checkpoint must be a mapping")
    if payload.get("schema_version") != VMC_OPTIMIZER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported VMC optimizer checkpoint schema")
    if payload.get("method") != "correlated_exterior_coordinate_vmc_adam":
        raise ValueError("unexpected VMC optimizer checkpoint method")
    return payload


def _checkpoint_payload(
    config: CorrelatedExteriorVMCOptimizerConfig,
    initialization_sha256: str,
    exponents_sha256: str,
    completed_steps: int,
    raw_orbitals: torch.Tensor,
    amplitudes: torch.Tensor,
    positions: torch.Tensor,
    generator: torch.Generator,
    raw_first_moment: torch.Tensor,
    raw_second_moment: torch.Tensor,
    amplitude_first_moment: torch.Tensor,
    amplitude_second_moment: torch.Tensor,
    history: list[dict[str, Any]],
    accepted_proposals: int,
    total_proposals: int,
) -> dict[str, Any]:
    return {
        "schema_version": VMC_OPTIMIZER_CHECKPOINT_SCHEMA_VERSION,
        "method": "correlated_exterior_coordinate_vmc_adam",
        "config": asdict(config),
        "initialization_sha256": initialization_sha256,
        "exponents_sha256": exponents_sha256,
        "completed_steps": completed_steps,
        "raw_orbitals": raw_orbitals.detach().clone(),
        "amplitudes": amplitudes.detach().clone(),
        "positions": positions.detach().clone(),
        "generator_state": generator.get_state(),
        "raw_first_moment": raw_first_moment.detach().clone(),
        "raw_second_moment": raw_second_moment.detach().clone(),
        "amplitude_first_moment": amplitude_first_moment.detach().clone(),
        "amplitude_second_moment": amplitude_second_moment.detach().clone(),
        "history": history,
        "accepted_proposals": accepted_proposals,
        "total_proposals": total_proposals,
    }


def run_correlated_exterior_vmc_optimization(
    config: CorrelatedExteriorVMCOptimizerConfig,
    initial_raw_orbitals: torch.Tensor,
    initial_amplitudes: torch.Tensor,
    exponents: torch.Tensor,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    max_steps_this_call: int | None = None,
) -> dict[str, Any]:
    """Optimize one state with exact checkpoint/resume trajectory recovery."""

    config.validate()
    if initial_raw_orbitals.ndim != 2 or initial_raw_orbitals.shape[1] != config.particles:
        raise ValueError("initial carrier must have shape (D,particles)")
    if initial_raw_orbitals.dtype != torch.float64:
        raise TypeError("initial carrier must use float64")
    if initial_amplitudes.ndim != 1 or exponents.ndim != 1:
        raise ValueError("amplitudes and exponents must be vectors")
    if initial_amplitudes.shape != exponents.shape:
        raise ValueError("amplitudes and exponents must have equal shape")
    if initial_amplitudes.dtype != torch.float64 or exponents.dtype != torch.float64:
        raise TypeError("correlator tensors must use float64")
    if exponents.numel() and bool(torch.any(exponents <= 0)):
        raise ValueError("correlator exponents must be positive")
    if max_steps_this_call is not None and max_steps_this_call < 1:
        raise ValueError("max_steps_this_call must be positive")

    initialization_sha256 = _tensor_state_sha256(
        initial_raw_orbitals, initial_amplitudes
    )
    exponents_sha256 = _tensor_state_sha256(exponents)
    generator = torch.Generator().manual_seed(config.seed)
    if resume:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise ValueError("resume requires an existing optimizer checkpoint")
        payload = _load_optimizer_checkpoint(checkpoint_path)
        if payload["config"] != asdict(config):
            raise ValueError("VMC optimizer checkpoint config mismatch")
        if payload["initialization_sha256"] != initialization_sha256:
            raise ValueError("VMC optimizer initialization mismatch")
        if payload["exponents_sha256"] != exponents_sha256:
            raise ValueError("VMC optimizer exponents mismatch")
        completed_steps = int(payload["completed_steps"])
        raw = payload["raw_orbitals"].clone()
        amplitudes = payload["amplitudes"].clone()
        positions = payload["positions"].clone()
        generator.set_state(payload["generator_state"])
        raw_first = payload["raw_first_moment"].clone()
        raw_second = payload["raw_second_moment"].clone()
        amplitude_first = payload["amplitude_first_moment"].clone()
        amplitude_second = payload["amplitude_second_moment"].clone()
        history = list(payload["history"])
        accepted = int(payload["accepted_proposals"])
        proposed = int(payload["total_proposals"])
    else:
        completed_steps = 0
        raw = canonical_exterior_carrier(initial_raw_orbitals.detach().clone())
        amplitudes = initial_amplitudes.detach().clone()
        positions = torch.randn(
            (config.chains, config.particles),
            generator=generator,
            dtype=torch.float64,
        )
        raw_first = torch.zeros_like(raw)
        raw_second = torch.zeros_like(raw)
        amplitude_first = torch.zeros_like(amplitudes)
        amplitude_second = torch.zeros_like(amplitudes)
        history = []
        accepted = 0
        proposed = 0

    remaining = config.steps - completed_steps
    admitted = remaining if max_steps_this_call is None else min(
        remaining, max_steps_this_call
    )
    estimator_config = config.estimator_config()
    for local_index in range(admitted):
        step = completed_steps + local_index
        orbitals = canonical_exterior_carrier(raw)
        log_probability = _log_probability(
            orbitals, amplitudes, exponents, positions
        )
        equilibration = (
            config.burn_in_sweeps if step == 0 else config.rethermalization_sweeps
        )
        step_accepted = 0
        step_proposed = 0
        for _ in range(equilibration):
            positions, log_probability, count, attempts = _metropolis_sweep(
                positions,
                log_probability,
                orbitals,
                amplitudes,
                exponents,
                config.proposal_scale,
                generator,
            )
            step_accepted += count
            step_proposed += attempts
        retained = []
        for _ in range(config.samples_per_chain):
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
                step_accepted += count
                step_proposed += attempts
            retained.append(positions.clone())
        samples = torch.stack(retained)
        observables = vmc_observables(
            estimator_config, orbitals, amplitudes, exponents, samples
        )
        gradient = vmc_energy_gradient(
            estimator_config, raw, amplitudes, exponents, samples
        )
        raw_gradient = gradient["orbital_gradient"]
        amplitude_gradient = gradient["amplitude_gradient"]
        gradient_norm = torch.sqrt(
            torch.sum(raw_gradient.square()) + torch.sum(amplitude_gradient.square())
        )
        clip_scale = min(
            1.0,
            config.gradient_clip_norm
            / max(float(gradient_norm), torch.finfo(torch.float64).tiny),
        )
        raw_gradient = raw_gradient * clip_scale
        amplitude_gradient = amplitude_gradient * clip_scale
        adam_step = step + 1
        raw_first = config.adam_beta1 * raw_first + (1 - config.adam_beta1) * raw_gradient
        raw_second = config.adam_beta2 * raw_second + (
            1 - config.adam_beta2
        ) * raw_gradient.square()
        amplitude_first = config.adam_beta1 * amplitude_first + (
            1 - config.adam_beta1
        ) * amplitude_gradient
        amplitude_second = config.adam_beta2 * amplitude_second + (
            1 - config.adam_beta2
        ) * amplitude_gradient.square()
        raw_first_corrected = raw_first / (1 - config.adam_beta1**adam_step)
        raw_second_corrected = raw_second / (1 - config.adam_beta2**adam_step)
        amplitude_first_corrected = amplitude_first / (
            1 - config.adam_beta1**adam_step
        )
        amplitude_second_corrected = amplitude_second / (
            1 - config.adam_beta2**adam_step
        )
        learning_rate = _learning_rate(config, step)
        raw = canonical_exterior_carrier(
            raw
            - learning_rate
            * raw_first_corrected
            / (torch.sqrt(raw_second_corrected) + config.adam_epsilon)
        ).detach()
        amplitudes = torch.clamp(
            amplitudes
            - learning_rate
            * amplitude_first_corrected
            / (torch.sqrt(amplitude_second_corrected) + config.adam_epsilon),
            min=-config.amplitude_bound,
            max=config.amplitude_bound,
        ).detach()
        accepted += step_accepted
        proposed += step_proposed
        history.append(
            {
                "step": adam_step,
                "energy": observables["energy"],
                "energy_variance": observables["energy_variance"],
                "energy_standard_error": observables["energy_standard_error"],
                "acceptance_rate": step_accepted / step_proposed,
                "effective_sample_size": observables["effective_sample_size"],
                "rhat": observables["rhat"],
                "antisymmetry_residual": observables["symmetry"][
                    "antisymmetry_residual"
                ],
                "correlator_symmetry_residual": observables["symmetry"][
                    "correlator_symmetry_residual"
                ],
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
                "gradient_clip_scale": clip_scale,
            }
        )
        completed_count = adam_step
        if checkpoint_path is not None and (
            completed_count % config.checkpoint_every == 0
            or completed_count == config.steps
        ):
            _save_checkpoint(
                checkpoint_path,
                _checkpoint_payload(
                    config,
                    initialization_sha256,
                    exponents_sha256,
                    completed_count,
                    raw,
                    amplitudes,
                    positions,
                    generator,
                    raw_first,
                    raw_second,
                    amplitude_first,
                    amplitude_second,
                    history,
                    accepted,
                    proposed,
                ),
            )

    completed_steps += admitted
    payload = _checkpoint_payload(
        config,
        initialization_sha256,
        exponents_sha256,
        completed_steps,
        raw,
        amplitudes,
        positions,
        generator,
        raw_first,
        raw_second,
        amplitude_first,
        amplitude_second,
        history,
        accepted,
        proposed,
    )
    if checkpoint_path is not None:
        _save_checkpoint(checkpoint_path, payload)
    return {
        "completed": completed_steps == config.steps,
        "completed_steps": completed_steps,
        "raw_orbitals": raw,
        "orbitals": canonical_exterior_carrier(raw),
        "amplitudes": amplitudes,
        "exponents": exponents.detach().clone(),
        "history": history,
        "accepted_proposals": accepted,
        "total_proposals": proposed,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "checkpoint_path": checkpoint_path.as_posix() if checkpoint_path else None,
        "materialization": {
            "D_to_the_N_tensor": False,
            "full_alternating_coefficient_tensor": False,
            "virtual_paths": 0,
        },
    }


__all__ = [
    "VMC_OPTIMIZER_CHECKPOINT_SCHEMA_VERSION",
    "CorrelatedExteriorVMCOptimizerConfig",
    "run_correlated_exterior_vmc_optimization",
]
