"""Small ordered-sector functional MPS comparator using latticeTN.

The state is stored on particle-coordinate sites but only strictly increasing
coordinate tuples are physical.  Dense gathering is retained for truth and AD
checks; this module is not yet a scalable constrained-MPO implementation.
"""

from __future__ import annotations

import itertools

import torch

from femps.ordered_sector import ordered_configurations


def ordered_values_to_particle_tensor(
    values: torch.Tensor, local_dimension: int, particles: int
) -> torch.Tensor:
    """Scatter ordered-sector values into a zero-padded ``D**N`` tensor."""

    configurations = ordered_configurations(local_dimension, particles)
    if values.ndim != 1 or values.numel() != len(configurations):
        raise ValueError("values must contain one amplitude per ordered configuration")
    tensor = torch.zeros(
        (local_dimension,) * particles, dtype=values.dtype, device=values.device
    )
    tensor[tuple(zip(*configurations, strict=True))] = values
    return tensor


def particle_tensor_to_mps_tensors(
    tensor: torch.Tensor,
    *,
    max_bond: int | None = None,
) -> tuple[list[torch.Tensor], tuple[int, ...], torch.Tensor]:
    """Apply sequential SVD and return cores, retained ranks, and discarded norm."""

    if tensor.ndim < 1 or len(set(tensor.shape)) != 1:
        raise ValueError("tensor must have equal particle-coordinate dimensions")
    if max_bond is not None and max_bond < 1:
        raise ValueError("max_bond must be positive")
    particles = tensor.ndim
    dimension = tensor.shape[0]
    remainder = tensor
    left_rank = 1
    cores = []
    ranks = []
    discarded_squared = torch.zeros((), dtype=tensor.real.dtype, device=tensor.device)
    for site in range(particles - 1):
        matrix = remainder.reshape(left_rank * dimension, -1)
        left, singular_values, right = torch.linalg.svd(matrix, full_matrices=False)
        if singular_values.numel() == 0 or singular_values[0] == 0:
            numerical_rank = 1
        else:
            tolerance = (
                max(matrix.shape)
                * torch.finfo(singular_values.dtype).eps
                * singular_values[0]
            )
            numerical_rank = max(1, int(torch.count_nonzero(singular_values > tolerance)))
        retained_rank = numerical_rank
        if max_bond is not None:
            retained_rank = min(retained_rank, max_bond)
        discarded_squared = discarded_squared + torch.sum(
            singular_values[retained_rank:] ** 2
        )
        cores.append(left[:, :retained_rank].reshape(left_rank, dimension, retained_rank))
        remainder = singular_values[:retained_rank, None] * right[:retained_rank]
        left_rank = retained_rank
        ranks.append(retained_rank)
    cores.append(remainder.reshape(left_rank, dimension, 1))
    return cores, tuple(ranks), torch.sqrt(discarded_squared)


def ordered_sector_functional_mps(
    values: torch.Tensor,
    local_dimension: int,
    particles: int,
    *,
    max_bond: int | None = None,
    requires_grad: bool = True,
):
    """Return a latticeTN MPS and exact/pre-truncation diagnostics."""

    try:
        from latticetn.mps import MPS
    except ImportError as exc:
        raise ImportError(
            "latticeTN is required; install the sibling repository with "
            "`python -m pip install -e ../latticeTN`"
        ) from exc

    tensor = ordered_values_to_particle_tensor(values, local_dimension, particles)
    cores, ranks, discarded_norm = particle_tensor_to_mps_tensors(
        tensor, max_bond=max_bond
    )
    mps = MPS.from_tensors(
        cores,
        dtype=values.dtype,
        device=values.device,
        requires_grad=requires_grad,
    )
    return mps, ranks, discarded_norm


def ordered_values_from_mps(mps) -> torch.Tensor:
    """Gather strictly increasing amplitudes from a latticeTN particle MPS."""

    configurations = ordered_configurations(mps.dim, mps.N)
    tensor = mps.to_dense().reshape((mps.dim,) * mps.N)
    return tensor[tuple(zip(*configurations, strict=True))]


def ordered_sector_dense_energy_from_mps(
    mps,
    ordered_hamiltonian: torch.Tensor,
) -> torch.Tensor:
    """Differentiable truth energy after gathering the ordered amplitudes."""

    values = ordered_values_from_mps(mps)
    if ordered_hamiltonian.shape != (values.numel(), values.numel()):
        raise ValueError("ordered_hamiltonian has the wrong shape")
    norm = torch.vdot(values, values)
    if norm == 0:
        raise ValueError("ordered component of the MPS is zero")
    return (torch.vdot(values, ordered_hamiltonian @ values) / norm).real
