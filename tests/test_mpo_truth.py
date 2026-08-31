import pytest
import torch

from femps.baselines.ordered_continuous_mpo import (
    ordered_continuous_soft_coulomb_hamiltonian_mpo,
)
from femps.benchmarks.mpo_truth import mpo_product_basis_matvec


def test_product_basis_mpo_matvec_matches_dense_operator() -> None:
    mpo = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        3, 3, 4.0, 4, interaction_quadrature_order=48
    )
    generator = torch.Generator().manual_seed(1600)
    vector = torch.randn(3**3, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(
        mpo_product_basis_matvec(mpo, vector),
        mpo.to_dense() @ vector,
        atol=1e-14,
        rtol=1e-14,
    )


def test_product_basis_mpo_matvec_validates_vector() -> None:
    mpo = ordered_continuous_soft_coulomb_hamiltonian_mpo(
        2, 2, 3.0, 2, interaction_quadrature_order=24
    )
    with pytest.raises(ValueError, match="exactly"):
        mpo_product_basis_matvec(mpo, torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="dtype"):
        mpo_product_basis_matvec(mpo, torch.ones(4, dtype=torch.float32))
