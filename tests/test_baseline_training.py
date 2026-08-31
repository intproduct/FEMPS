import pytest

from femps.baselines.training import BaselineConfig


def test_baseline_config_rejects_invalid_learning_rate_schedule():
    with pytest.raises(ValueError, match="final_learning_rate"):
        BaselineConfig(learning_rate=1e-3, final_learning_rate=1e-2).validate()


def test_baseline_config_rejects_unbounded_chain():
    with pytest.raises(ValueError, match="positive definite"):
        BaselineConfig(num_oscillators=16, gamma=0.7).validate()

