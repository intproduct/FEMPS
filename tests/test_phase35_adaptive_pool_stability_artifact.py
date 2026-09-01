from pathlib import Path

from scripts.verify_phase35_adaptive_pool_stability import verify_artifact


def test_phase35_adaptive_pool_stability_artifact() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase35_adaptive_pool_stability.json")
    )
    assert result["verified"]
    assert result["K6_energy_spread"] < 5e-6
    assert result["maximum_K6_error_vs_CI"] < 3.8e-5
    assert result["maximum_K6_variance"] < 4.6e-4
    assert result["minimum_K4_to_K6_improvement"] > 6.7e-5
    assert not result["automatic_stopping_rule_admitted"]
