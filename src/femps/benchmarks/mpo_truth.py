"""Small product-basis truth helpers for native MPO audits."""

from __future__ import annotations

import string

import torch


def mpo_product_basis_matvec(mpo, vector: torch.Tensor) -> torch.Tensor:
    """Apply an MPO without constructing its product-basis matrix.

    The vector itself is deliberately dense: this helper is for independently
    bounded truth audits, not production training.  The Hamiltonian matrix of
    size ``d**N`` squared is never materialized.
    """

    sites = mpo.length
    labels = list(string.ascii_letters)
    if 3 * sites + 1 > len(labels):
        raise ValueError("not enough einsum labels for this MPO")
    if vector.ndim != 1 or vector.numel() != mpo.dim**sites:
        raise ValueError("vector must have exactly mpo.dim**mpo.length entries")
    if vector.dtype != mpo.dtype or vector.device != mpo.tensors[0].device:
        raise ValueError("vector and MPO must share dtype and device")
    inputs = labels[:sites]
    outputs = labels[sites : 2 * sites]
    virtual = labels[2 * sites : 3 * sites + 1]
    operands = ["".join(inputs)]
    operands.extend(
        virtual[site]
        + virtual[site + 1]
        + inputs[site]
        + outputs[site]
        for site in range(sites)
    )
    equation = ",".join(operands) + "->" + "".join(outputs)
    state = vector.reshape((mpo.dim,) * sites)
    return torch.einsum(equation, state, *mpo.tensors).reshape(-1)
