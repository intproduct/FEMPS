import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("latticetn")

from femps.algorithms.ordered_continuous_training import (
    OrderedContinuousTrainingConfig,
    random_uniform_functional_mps,
    train_ordered_continuous_mps,
)


def test_random_functional_mps_uses_local_dimension_geometry() -> None:
    mps = random_uniform_functional_mps(
        3, 5, 9, seed=4
    )
    assert [tuple(tensor.shape) for tensor in mps.tensors] == [
        (1, 5, 5),
        (5, 5, 5),
        (5, 5, 1),
    ]
    assert abs(float(mps.norm_sq().detach()) - 1) < 2e-13


def test_small_continuous_native_training_decreases_energy() -> None:
    config = OrderedContinuousTrainingConfig(
        particles=2,
        basis_order=5,
        distance_length=6.0,
        interaction_degree=10,
        interaction_quadrature_order=80,
        bond_dimension=5,
        steps=80,
        learning_rate=0.03,
        seed=23,
        projection="tensor_norm",
    )
    _, diagnostics = train_ordered_continuous_mps(config)
    assert diagnostics["final_energy"] < diagnostics["initial_energy"] - 0.2
    assert diagnostics["native_training_materializes_product_tensor"] is False
    assert abs(diagnostics["physical_norm_after_projection"] - 1) < 2e-13


def test_small_unbounded_fourier_training_decreases_energy() -> None:
    config = OrderedContinuousTrainingConfig(
        particles=2,
        basis_order=5,
        distance_length=1.1,
        distance_basis="odd_hermite",
        interaction_method="fourier_bessel",
        fourier_order=48,
        interaction_quadrature_order=96,
        bond_dimension=5,
        steps=60,
        learning_rate=0.03,
        seed=24,
        projection="tensor_norm",
    )
    _, diagnostics = train_ordered_continuous_mps(config)
    assert diagnostics["final_energy"] < diagnostics["initial_energy"] - 0.2
    assert diagnostics["native_training_materializes_product_tensor"] is False
    assert abs(diagnostics["physical_norm_after_projection"] - 1) < 2e-13


def test_fourier_training_rejects_finite_interval_basis() -> None:
    config = OrderedContinuousTrainingConfig(
        particles=2,
        basis_order=3,
        distance_basis="dirichlet_sine",
        interaction_method="fourier_bessel",
        steps=1,
    )
    with pytest.raises(ValueError, match="requires odd_hermite"):
        train_ordered_continuous_mps(config)


def test_fourier_training_records_mpo_compression() -> None:
    config = OrderedContinuousTrainingConfig(
        particles=3,
        basis_order=3,
        distance_length=0.9,
        distance_basis="odd_hermite",
        interaction_method="fourier_bessel",
        fourier_order=8,
        interaction_quadrature_order=64,
        mpo_max_bond=6,
        bond_dimension=3,
        steps=2,
        learning_rate=0.01,
        seed=25,
        projection="tensor_norm",
    )
    _, diagnostics = train_ordered_continuous_mps(config)
    assert diagnostics["uncompressed_mpo_max_bond"] > 6
    assert diagnostics["mpo_max_bond"] == 6
    assert diagnostics["mpo_compression_ranks"] == [6, 6]
    assert diagnostics["mpo_compression_local_discarded_norm"] > 0
