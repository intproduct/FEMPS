from pathlib import Path

from scripts.verify_phase36_public_adaptive_solver import verify_artifact


def test_phase36_public_adaptive_solver_artifact() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase36_public_adaptive_solver.json")
    )
    assert result["verified"]
    assert result["maximum_energy_difference_vs_phase35"] == 0.0
    assert result["final_error_vs_CI"] < 3.3e-5
    assert result["final_variance"] < 4e-4
    assert result["automatic_stopping_rule"] == "not_admitted"
