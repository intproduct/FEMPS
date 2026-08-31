"""Hard-charge AD optimization for ordered-distance functional MPS."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from femps.baselines.ordered_distance_mpo import (
    gap_soft_coulomb_hamiltonian_mpo,
)


def bounded_composition_count(parts: int, total: int, maximum: int) -> int:
    """Count length-``parts`` bounded weak compositions exactly."""

    if parts < 0 or total < 0 or maximum < 0:
        return 0
    counts = [0] * (total + 1)
    counts[0] = 1
    for _ in range(parts):
        next_counts = [0] * (total + 1)
        for accumulated, count in enumerate(counts):
            if not count:
                continue
            for local in range(min(maximum, total - accumulated) + 1):
                next_counts[accumulated + local] += count
        counts = next_counts
    return counts[total]


def gap_bond_charge_labels(
    sites: int,
    local_maximum: int,
    total_charge: int,
    multiplicity_per_charge: int,
) -> list[tuple[int, ...]]:
    """Return repeated cumulative-charge labels for every MPS bond."""

    if (
        sites < 1
        or local_maximum < 0
        or total_charge < 0
        or multiplicity_per_charge < 1
        or total_charge > sites * local_maximum
    ):
        raise ValueError("invalid sites, local maximum, total charge, or multiplicity")
    labels = []
    for bond in range(sites + 1):
        minimum = max(0, total_charge - (sites - bond) * local_maximum)
        maximum = min(total_charge, bond * local_maximum)
        bond_labels = []
        for charge in range(minimum, maximum + 1):
            left_ways = bounded_composition_count(bond, charge, local_maximum)
            right_ways = bounded_composition_count(
                sites - bond, total_charge - charge, local_maximum
            )
            multiplicity = min(
                multiplicity_per_charge, left_ways, right_ways
            )
            bond_labels.extend([charge] * multiplicity)
        labels.append(tuple(bond_labels))
    if labels[0] != (0,) or labels[-1] != (total_charge,):
        raise AssertionError("hard-charge boundary labels are inconsistent")
    return labels


def gap_charge_masks(
    bond_labels: list[tuple[int, ...]],
    local_dimension: int,
    *,
    device: torch.device | str | None = None,
) -> list[torch.Tensor]:
    """Return masks enforcing ``q_right=q_left+q_local``."""

    masks = []
    for left_labels, right_labels in zip(bond_labels[:-1], bond_labels[1:]):
        mask = torch.zeros(
            len(left_labels),
            local_dimension,
            len(right_labels),
            dtype=torch.bool,
            device=device,
        )
        for left_index, left_charge in enumerate(left_labels):
            for local_charge in range(local_dimension):
                target = left_charge + local_charge
                for right_index, right_charge in enumerate(right_labels):
                    if right_charge == target:
                        mask[left_index, local_charge, right_index] = True
        masks.append(mask)
    return masks


def apply_gap_masks_(mps, masks: list[torch.Tensor]) -> None:
    """Zero forbidden tensor entries in place."""

    with torch.no_grad():
        for tensor, mask in zip(mps.tensors, masks, strict=True):
            tensor.mul_(mask.to(dtype=tensor.dtype, device=tensor.device))


def zero_forbidden_gap_gradients_(mps, masks: list[torch.Tensor]) -> None:
    """Zero gradients outside the hard charge blocks."""

    for tensor, mask in zip(mps.tensors, masks, strict=True):
        if tensor.grad is not None:
            tensor.grad.mul_(mask.to(dtype=tensor.grad.dtype, device=tensor.grad.device))


def random_gap_charge_mps(
    grid_points: int,
    particles: int,
    gap_cutoff: int,
    multiplicity_per_charge: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
):
    """Create a random latticeTN MPS exactly inside the finite-box gap sector."""

    try:
        from latticetn.mps import MPS
    except ImportError as exc:
        raise ImportError("latticeTN is required for ordered-distance training") from exc
    holes = grid_points - particles
    sites = particles + 1
    if not 0 <= gap_cutoff <= holes:
        raise ValueError("gap_cutoff must satisfy 0 <= gap_cutoff <= L-N")
    labels = gap_bond_charge_labels(
        sites, gap_cutoff, holes, multiplicity_per_charge
    )
    masks = gap_charge_masks(labels, gap_cutoff + 1, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    tensors = [
        torch.randn(mask.shape, dtype=dtype, device=device, generator=generator)
        / math.sqrt(max(1, mask.shape[1] * mask.shape[2]))
        for mask in masks
    ]
    mps = MPS.from_tensors(
        tensors, dtype=dtype, device=device, requires_grad=True
    )
    apply_gap_masks_(mps, masks)
    norm = mps.norm_sq()
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError("random hard-charge MPS has invalid norm")
    with torch.no_grad():
        mps.tensors[0].div_(torch.sqrt(norm))
    return mps, masks, labels


@dataclass(frozen=True)
class OrderedDistanceTrainingConfig:
    grid_points: int = 8
    particles: int = 4
    spacing: float = 0.7
    gap_cutoff: int = 4
    multiplicity_per_charge: int = 4
    steps: int = 1000
    learning_rate: float = 0.02
    seed: int = 701
    gradient_clip: float = 10.0
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


def train_ordered_distance_mps(
    config: OrderedDistanceTrainingConfig,
) -> tuple[object, dict[str, object]]:
    """Optimize a random hard-charge MPS using only native MPS/MPO energy."""

    device = torch.device(config.device)
    mps, masks, labels = random_gap_charge_mps(
        config.grid_points,
        config.particles,
        config.gap_cutoff,
        config.multiplicity_per_charge,
        seed=config.seed,
        dtype=config.dtype,
        device=device,
    )
    mpo = gap_soft_coulomb_hamiltonian_mpo(
        config.grid_points,
        config.particles,
        config.spacing,
        gap_cutoff=config.gap_cutoff,
        dtype=config.dtype,
        device=device,
    )
    optimizer = torch.optim.Adam(mps.parameters(), lr=config.learning_rate)
    history = []
    best_energy = math.inf
    best_tensors = None
    for step in range(config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        energy = mps.energy_with_MPO(mpo)
        if not torch.isfinite(energy):
            raise RuntimeError(f"nonfinite energy at step {step}")
        value = float(energy.detach().cpu())
        if value < best_energy:
            best_energy = value
            best_tensors = [tensor.detach().clone() for tensor in mps.tensors]
        if step in {0, config.steps} or step % max(1, config.steps // 20) == 0:
            history.append({"step": step, "energy": value})
        if step == config.steps:
            break
        energy.backward()
        zero_forbidden_gap_gradients_(mps, masks)
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(mps.parameters(), config.gradient_clip)
        optimizer.step()
        apply_gap_masks_(mps, masks)
        if (step + 1) % 20 == 0:
            norm = mps.norm_sq()
            with torch.no_grad():
                mps.tensors[0].div_(torch.sqrt(norm))
    if best_tensors is None:
        raise AssertionError("training did not record a state")
    with torch.no_grad():
        for target, source in zip(mps.tensors, best_tensors, strict=True):
            target.copy_(source)
    final_energy = float(mps.energy_with_MPO(mpo).detach().cpu())
    diagnostics: dict[str, object] = {
        "initial_energy": history[0]["energy"],
        "best_energy": best_energy,
        "final_energy": final_energy,
        "history": history,
        "bond_dimensions": [len(charges) for charges in labels],
        "charge_labels": [list(charges) for charges in labels],
        "max_forbidden_parameter": max(
            float(torch.max(torch.abs(tensor.detach()[~mask])))
            if torch.any(~mask) else 0.0
            for tensor, mask in zip(mps.tensors, masks, strict=True)
        ),
        "native_only": True,
    }
    return mps, diagnostics
