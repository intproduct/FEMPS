"""Blind latticeTN AD training for continuous ordered functional MPS."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from femps.baselines.ordered_continuous_mpo import (
    ordered_continuous_soft_coulomb_hamiltonian_mpo,
)


def random_uniform_functional_mps(
    sites: int,
    local_dimension: int,
    bond_dimension: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
):
    """Create a geometry-capped random MPS for arbitrary local dimension."""

    if sites < 1 or local_dimension < 1 or bond_dimension < 1:
        raise ValueError("sites, local dimension, and bond dimension must be positive")
    try:
        from latticetn.mps import MPS
    except ImportError as exc:
        raise ImportError("latticeTN is required for continuous training") from exc
    bonds = [
        min(bond_dimension, local_dimension ** min(site, sites - site))
        for site in range(sites + 1)
    ]
    generator = torch.Generator(device=device).manual_seed(seed)
    tensors = [
        torch.randn(
            bonds[site],
            local_dimension,
            bonds[site + 1],
            dtype=dtype,
            device=device,
            generator=generator,
        )
        / (local_dimension * bonds[site] * bonds[site + 1]) ** 0.5
        for site in range(sites)
    ]
    mps = MPS.from_tensors(
        tensors,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    norm = mps.norm_sq()
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError("random functional MPS has an invalid norm")
    with torch.no_grad():
        mps.tensors[0].div_(torch.sqrt(norm))
    return mps


@dataclass(frozen=True)
class OrderedContinuousTrainingConfig:
    particles: int = 2
    basis_order: int = 12
    distance_length: float = 9.0
    distance_basis: str = "dirichlet_sine"
    interaction_degree: int = 20
    interaction_quadrature_order: int = 160
    bond_dimension: int = 12
    coupling: float = 1.0
    softening: float = 1.0
    steps: int = 800
    learning_rate: float = 0.02
    seed: int = 1601
    optimizer: str = "adam"
    projection: str = "canonical"
    canonical_interval: int = 100
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


def train_ordered_continuous_mps(
    config: OrderedContinuousTrainingConfig,
) -> tuple[object, dict[str, object]]:
    """Train using only latticeTN native MPS/MPO contractions and AD."""

    try:
        from latticetn.ad_variational import ADVariationalMPS, train_ad_mps
    except ImportError as exc:
        raise ImportError("latticeTN is required for continuous training") from exc
    device = torch.device(config.device)
    mps = random_uniform_functional_mps(
        config.particles,
        config.basis_order,
        config.bond_dimension,
        seed=config.seed,
        dtype=config.dtype,
        device=device,
    )
    mpo = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        config.particles,
        config.basis_order,
        config.distance_length,
        config.interaction_degree,
        distance_basis=config.distance_basis,
        coupling=config.coupling,
        softening=config.softening,
        interaction_quadrature_order=config.interaction_quadrature_order,
        dtype=config.dtype,
        device=device,
    )
    model = ADVariationalMPS(mps, mpo)
    diagnostics = train_ad_mps(
        model,
        num_steps=config.steps,
        lr=config.learning_rate,
        optimizer=config.optimizer,
        projection=config.projection,
        record_every=max(1, config.steps // 20),
        canonical_interval=config.canonical_interval,
        verbose=False,
    )
    diagnostics.update(
        {
            "seed": config.seed,
            "mps_parameter_count": sum(
                tensor.numel() for tensor in model.mps.tensors
            ),
            "mpo_max_bond": max(
                max(tensor.shape[:2]) for tensor in mpo.tensors
            ),
            "native_training_materializes_product_tensor": False,
        }
    )
    return model.mps, diagnostics
