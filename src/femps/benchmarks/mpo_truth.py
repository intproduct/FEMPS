"""Small product-basis truth helpers for native MPO audits."""

from __future__ import annotations

import string
import time

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


def lowest_mpo_eigenpair(
    mpo,
    *,
    tolerance: float = 2e-10,
    maximum_iterations: int = 1200,
    seed: int = 1610,
    initial_vector: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor, dict[str, float | int | bool | str]]:
    """Return the lowest eigenpair through a CPU product-basis Lanczos audit.

    The dense state vector is bounded by ``dim**length`` but the Hamiltonian
    matrix is never materialized. SciPy remains an optional benchmark-only
    dependency and is imported lazily.
    """

    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh

    if any(tensor.device.type != "cpu" for tensor in mpo.tensors):
        raise ValueError("the independent Lanczos audit is CPU-only")
    if mpo.dtype != torch.float64:
        raise ValueError("the independent Lanczos audit currently requires float64")
    dimension = mpo.dim**mpo.length
    calls = 0

    def matvec(vector: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        tensor = torch.from_numpy(np.asarray(vector, dtype=np.float64))
        return mpo_product_basis_matvec(mpo, tensor).detach().numpy()

    operator = LinearOperator(
        (dimension, dimension), matvec=matvec, rmatvec=matvec, dtype=np.float64
    )
    if initial_vector is None:
        initial = np.random.default_rng(seed).normal(size=dimension)
        initialization = "seeded_random"
    else:
        if (
            initial_vector.ndim != 1
            or initial_vector.numel() != dimension
            or initial_vector.dtype != torch.float64
            or initial_vector.device.type != "cpu"
        ):
            raise ValueError(
                "initial_vector must be a CPU float64 vector of dim**length entries"
            )
        initial = initial_vector.detach().numpy()
        initialization = "provided_post_training_vector"
    started = time.perf_counter()
    values, vectors = eigsh(
        operator,
        k=1,
        which="SA",
        v0=initial,
        tol=tolerance,
        maxiter=maximum_iterations,
    )
    elapsed = time.perf_counter() - started
    eigenvector = torch.from_numpy(vectors[:, 0].copy())
    residual = torch.linalg.vector_norm(
        mpo_product_basis_matvec(mpo, eigenvector) - values[0] * eigenvector
    )
    diagnostics: dict[str, float | int | bool | str] = {
        "product_basis_dimension": dimension,
        "matvec_calls": calls,
        "residual_norm": float(residual),
        "elapsed_seconds": elapsed,
        "dense_hamiltonian_materialized": False,
        "initialization": initialization,
    }
    return float(values[0]), eigenvector, diagnostics
