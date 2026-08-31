"""Continuum-safe initialization using the sibling latticeTN MPS class."""

from __future__ import annotations

import torch


def random_functional_mps(
    num_variables: int,
    basis_order: int,
    bond_dimension: int,
    *,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str = "cpu",
    seed: int = 0,
):
    """Create an open functional MPS with geometry-aware continuum bonds.

    The current upstream ``latticeTN.MPS`` constructor caps each bond using
    ``2**min(i, N-i)``, appropriate to spin-1/2 sites but not a functional
    basis with local dimension ``D``. Explicit tensors let FEMPS reuse the
    upstream class and contractions while using ``D**min(i, N-i)``.
    """
    if num_variables < 1 or basis_order < 1 or bond_dimension < 1:
        raise ValueError("num_variables, basis_order, and bond_dimension must be positive")

    try:
        from latticetn.mps import MPS
    except ImportError as exc:
        raise ImportError(
            "latticeTN is required; install the sibling repository with "
            "`python -m pip install -e ../latticeTN`"
        ) from exc

    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    bonds = [
        min(bond_dimension, basis_order ** min(i, num_variables - i))
        for i in range(num_variables + 1)
    ]
    tensors = [
        torch.randn(
            (bonds[i], basis_order, bonds[i + 1]),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        for i in range(num_variables)
    ]
    return MPS.from_tensors(tensors, dtype=dtype, device=device, requires_grad=True)

