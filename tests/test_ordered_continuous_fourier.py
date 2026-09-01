import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.integration
pytest.importorskip("latticetn")

from femps.basis.odd_hermite import odd_hermite_basis_values
from femps.baselines.ordered_continuous_fourier import (
    ordered_continuous_fourier_hamiltonian_compressed_mpo,
    ordered_continuous_fourier_hamiltonian_mpo,
    ordered_continuous_fourier_soft_coulomb_mpo,
    ordered_continuous_fourier_soft_coulomb_pair_mpo,
    soft_coulomb_fourier_sampled_error,
)
from femps.baselines.ordered_continuous_mpo import (
    ordered_continuous_noninteracting_mpo,
)
from femps.baselines.ordered_distance_mpo import compress_mpo


def _quadrature(order: int, scale: float, count: int = 500):
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(count)
    cutoff = scale * (math.sqrt(4 * order + 2) + 8)
    nodes = torch.from_numpy((raw_nodes + 1) * cutoff / 2)
    weights = torch.from_numpy(raw_weights * cutoff / 2)
    basis = odd_hermite_basis_values(order, nodes, scale)
    return nodes, weights, basis


def test_fourier_bessel_scalar_error_decreases_with_order() -> None:
    errors = [
        soft_coulomb_fourier_sampled_error(10.0, order)
        for order in (64, 96, 128)
    ]
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 1e-7


def test_one_gap_fourier_mpo_matches_direct_unbounded_projection() -> None:
    order = 5
    scale = 0.8
    nodes, weights, basis = _quadrature(order, scale)
    expected = torch.einsum(
        "xm,x,x,xn->mn",
        basis,
        weights,
        torch.rsqrt(nodes.square() + 1),
        basis,
    )
    errors = []
    for fourier_order in (64, 128):
        mpo = ordered_continuous_fourier_soft_coulomb_pair_mpo(
            2,
            order,
            scale,
            0,
            1,
            fourier_order,
            local_quadrature_order=160,
        )
        observed = mpo.to_dense().reshape(order, order, order, order)[
            0, :, 0, :
        ]
        errors.append(float(torch.linalg.matrix_norm(observed - expected)))
    assert errors[1] < errors[0]
    assert errors[-1] < 2e-7


def test_two_gap_rotation_mpo_matches_direct_unbounded_projection() -> None:
    order = 3
    scale = 0.8
    nodes, weights, basis = _quadrature(order, scale, 180)
    densities = torch.einsum("xm,xn->xmn", basis, basis)
    potential = torch.rsqrt(
        (nodes[:, None] + nodes[None, :]).square() + 1
    )
    relative = torch.einsum(
        "xmn,yab,xy,x,y->manb",
        densities,
        densities,
        potential,
        weights,
        weights,
    ).reshape(order**2, order**2)
    expected = torch.kron(torch.eye(order), relative)
    errors = []
    for fourier_order in (64, 128):
        mpo = ordered_continuous_fourier_soft_coulomb_pair_mpo(
            3,
            order,
            scale,
            0,
            2,
            fourier_order,
            local_quadrature_order=160,
        )
        observed = mpo.to_dense()
        torch.testing.assert_close(observed.mT, observed, atol=3e-13, rtol=3e-13)
        errors.append(float(torch.linalg.matrix_norm(observed - expected)))
        assert max(max(tensor.shape[:2]) for tensor in mpo.tensors) <= 2 * fourier_order
    assert errors[1] < errors[0]
    assert errors[-1] < 4e-7


def test_n2_unbounded_fourier_energy_matches_direct_projected_interaction() -> None:
    order = 6
    scale = 1.2
    nodes, weights, basis = _quadrature(order, scale)
    direct_potential = torch.einsum(
        "xm,x,x,xn->mn",
        basis,
        weights,
        torch.rsqrt(nodes.square() + 1),
        basis,
    )
    direct = ordered_continuous_noninteracting_mpo(
        2, order, scale, distance_basis="odd_hermite"
    ).to_dense() + torch.kron(torch.eye(order), direct_potential)
    expected_energy = torch.linalg.eigvalsh(direct)[0]
    errors = []
    for fourier_order in (64, 160):
        observed = ordered_continuous_fourier_hamiltonian_mpo(
            2,
            order,
            scale,
            fourier_order,
            local_quadrature_order=160,
        ).to_dense()
        errors.append(float(torch.linalg.eigvalsh(observed)[0] - expected_energy))
    assert 0 < errors[1] < errors[0]
    assert errors[-1] < 4e-8


@pytest.mark.parametrize("particles", [2, 3, 4, 5])
def test_compact_all_pair_mpo_matches_direct_pair_audit(particles: int) -> None:
    kwargs = dict(
        particles=particles,
        basis_order=2,
        distance_scale=0.9,
        fourier_order=8,
        local_quadrature_order=96,
    )
    compact = ordered_continuous_fourier_soft_coulomb_mpo(
        **kwargs, construction="compact"
    )
    direct = ordered_continuous_fourier_soft_coulomb_mpo(
        **kwargs, construction="direct_pairs"
    )
    torch.testing.assert_close(
        compact.to_dense(), direct.to_dense(), atol=2e-13, rtol=2e-13
    )
    if particles > 2:
        assert max(max(tensor.shape[:2]) for tensor in compact.tensors) == 32


@pytest.mark.parametrize(
    "particles,basis_order,fourier_order,maximum_bond",
    [(3, 3, 12, 8), (4, 3, 12, 12), (5, 2, 8, 12)],
)
def test_incremental_structured_compression_matches_raw_mpo_compression(
    particles: int,
    basis_order: int,
    fourier_order: int,
    maximum_bond: int,
) -> None:
    kwargs = dict(
        particles=particles,
        basis_order=basis_order,
        distance_scale=0.85,
        fourier_order=fourier_order,
        coupling=1.2,
        local_quadrature_order=96,
    )
    raw = ordered_continuous_fourier_hamiltonian_mpo(**kwargs)
    expected, expected_ranks, expected_discarded = compress_mpo(
        raw, maximum_bond
    )
    observed, diagnostics = (
        ordered_continuous_fourier_hamiltonian_compressed_mpo(
            **kwargs, maximum_bond=maximum_bond
        )
    )
    torch.testing.assert_close(
        observed.to_dense(), expected.to_dense(), atol=3e-12, rtol=3e-12
    )
    assert diagnostics["retained_ranks"] == expected_ranks
    torch.testing.assert_close(
        diagnostics["local_discarded_norm_not_global_certificate"],
        expected_discarded,
        atol=3e-11,
        rtol=3e-11,
    )
    assert diagnostics["dense_raw_fourier_bulk_materialized"] is False
    assert diagnostics["theoretical_raw_maximum_bond"] == max(
        max(tensor.shape[:2]) for tensor in raw.tensors
    )
    assert diagnostics["theoretical_raw_tensor_elements"] == sum(
        tensor.numel() for tensor in raw.tensors
    )
    assert diagnostics["maximum_intermediate_tensor_elements"] < sum(
        tensor.numel() for tensor in raw.tensors
    )


def test_multiscale_structured_hamiltonian_matches_materialized_compression() -> None:
    kwargs = dict(
        particles=4,
        basis_order=3,
        distance_scale=0.8,
        fourier_order=12,
        distance_basis="multiscale_odd_hermite",
        distance_scale_ratio=2.0,
        local_quadrature_order=96,
    )
    raw = ordered_continuous_fourier_hamiltonian_mpo(**kwargs)
    expected, _, _ = compress_mpo(raw, 12)
    observed, diagnostics = (
        ordered_continuous_fourier_hamiltonian_compressed_mpo(
            **kwargs, maximum_bond=12
        )
    )
    torch.testing.assert_close(
        observed.to_dense(), expected.to_dense(), atol=4e-12, rtol=4e-12
    )
    assert diagnostics["dense_raw_fourier_bulk_materialized"] is False
