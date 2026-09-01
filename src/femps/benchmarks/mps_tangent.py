"""Gauge-fixed physical MPS tangent directions for operator audits."""

from __future__ import annotations

import torch


def left_gauge_physical_tangent_directions(
    mps,
    *,
    directions_per_site: int = 2,
    seed: int = 1910,
):
    """Return a left-canonical state and normalized one-site tangents.

    For every nonfinal site, the flattened direction ``B`` obeys
    ``A^T B=0`` in the left-canonical gauge.  At the final site the state
    component is removed directly.  Each resulting many-body tangent state is
    normalized with the native MPS overlap, so downstream directional
    derivatives are physical rather than raw parameter-coordinate quantities.

    The current audit is deliberately float64-only.  Complex tangent spaces
    require an explicit real/complex Wirtinger convention before admission.
    """

    if directions_per_site < 1:
        raise ValueError("directions_per_site must be positive")
    if mps.dtype != torch.float64:
        raise ValueError("the gauge-fixed tangent audit requires float64")
    try:
        from latticetn.canonical import left_canonical
        from latticetn.mps import MPS
    except ImportError as exc:
        raise ImportError("latticeTN is required for MPS tangent audits") from exc

    canonical = left_canonical(mps)
    generator = torch.Generator(device=canonical.device).manual_seed(seed)
    directions = []
    for site, tensor in enumerate(canonical.tensors):
        for direction_index in range(directions_per_site):
            direction = torch.randn(
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
                generator=generator,
            )
            if site < canonical.N - 1:
                columns = tensor.reshape(-1, tensor.shape[2])
                tangent_columns = direction.reshape(-1, tensor.shape[2])
                tangent_columns = tangent_columns - columns @ (
                    columns.T @ tangent_columns
                )
                direction = tangent_columns.reshape_as(tensor)
                gauge_residual = torch.linalg.matrix_norm(
                    columns.T @ tangent_columns
                )
            else:
                overlap = torch.sum(tensor * direction)
                norm_squared = torch.sum(tensor.square())
                direction = direction - tensor * overlap / norm_squared
                gauge_residual = torch.abs(torch.sum(tensor * direction))

            tangent_tensors = list(canonical.tensors)
            tangent_tensors[site] = direction
            tangent_mps = MPS.from_tensors(
                tangent_tensors,
                dtype=canonical.dtype,
                device=canonical.device,
                requires_grad=False,
            )
            physical_norm = torch.sqrt(tangent_mps.norm_sq())
            if not torch.isfinite(physical_norm):
                raise RuntimeError("projected MPS tangent norm is not finite")
            if physical_norm <= 1e-12:
                # A full-column-rank canonical tensor can occupy the complete
                # local row space (for example the first geometry-capped
                # site).  Its left-gauge orthogonal complement is genuinely
                # zero-dimensional, so there is no one-site tangent block.
                continue
            direction = direction / physical_norm
            tangent_tensors[site] = direction
            normalized_tangent = MPS.from_tensors(
                tangent_tensors,
                dtype=canonical.dtype,
                device=canonical.device,
                requires_grad=False,
            )
            directions.append(
                {
                    "site": site,
                    "direction_index": direction_index,
                    "tensor": direction,
                    "pre_normalization_physical_norm": float(physical_norm),
                    "gauge_residual_before_physical_normalization": float(
                        gauge_residual
                    ),
                    "normalized_physical_norm": float(
                        torch.sqrt(normalized_tangent.norm_sq())
                    ),
                    "state_overlap_absolute_value": float(
                        torch.abs(canonical.overlap(normalized_tangent))
                    ),
                }
            )
    return canonical, directions


def mpo_energy_and_tangent_directional_derivatives(
    canonical_mps,
    mpo,
    directions,
) -> tuple[float, list[float], float]:
    """Return energy and derivatives along fixed gauge/physical directions."""

    try:
        from latticetn.mps import MPS
    except ImportError as exc:
        raise ImportError("latticeTN is required for MPS tangent audits") from exc
    probe = MPS.from_tensors(
        [tensor.detach().clone() for tensor in canonical_mps.tensors],
        dtype=canonical_mps.dtype,
        device=canonical_mps.device,
        requires_grad=True,
    )
    energy = probe.energy_with_MPO(mpo)
    energy.backward()
    derivatives = [
        float(
            torch.sum(
                probe.tensors[direction["site"]].grad
                * direction["tensor"]
            ).detach()
        )
        for direction in directions
    ]
    gradient_norm = torch.sqrt(
        sum(torch.sum(tensor.grad.square()) for tensor in probe.tensors)
    )
    return float(energy.detach()), derivatives, float(gradient_norm.detach())
