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
