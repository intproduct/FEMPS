from pathlib import Path

from scripts.verify_phase34_adaptive_k_growth import verify_artifact


def test_phase34_adaptive_k_growth_artifact() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase34_adaptive_k_growth.json")
    )
    assert result["verified"]
    assert result["K6_energy"] < result["K5_energy"] < result["K4_energy"]
    assert result["K6_energy"] < result["cold_K6_energy"]
    assert result["K6_error_vs_CI"] < 3.3e-5
    assert result["K6_variance"] < 4e-4
