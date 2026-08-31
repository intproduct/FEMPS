import pytest
import torch

pytestmark = pytest.mark.integration
pytest.importorskip("latticetn")

from femps.algorithms.ordered_distance_training import (
    OrderedDistanceTrainingConfig,
    gap_bond_charge_labels,
    train_ordered_distance_mps,
)
from femps.baselines.ordered_distance_mpo import gap_charge_projector_mpo


def test_gap_bond_labels_respect_reachability_and_multiplicity() -> None:
    labels = gap_bond_charge_labels(5, 4, 4, 3)
    assert labels[0] == (0,)
    assert labels[-1] == (4,)
    assert max(len(bond) for bond in labels) > 4
    assert all(bond == tuple(sorted(bond)) for bond in labels)


def test_native_hard_charge_training_decreases_energy_without_leakage() -> None:
    config = OrderedDistanceTrainingConfig(
        grid_points=6,
        particles=3,
        spacing=0.7,
        gap_cutoff=3,
        multiplicity_per_charge=2,
        steps=80,
        learning_rate=0.03,
        seed=17,
    )
    mps, diagnostics = train_ordered_distance_mps(config)
    assert diagnostics["best_energy"] < diagnostics["initial_energy"] - 0.1
    assert diagnostics["max_forbidden_parameter"] == 0
    projector = gap_charge_projector_mpo(6, 3)
    weight = (mps._expect_MPO(projector) / mps.overlap(mps)).real
    torch.testing.assert_close(
        weight, torch.ones((), dtype=torch.float64), atol=2e-13, rtol=2e-13
    )
