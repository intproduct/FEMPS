from pathlib import Path

from scripts.verify_phase32_n6_convergence import verify_artifact


def test_phase32_n6_convergence_artifact() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase32_n6_convergence.json")
    )
    assert result["verified"] is True
    assert len(result["K_energies"]) == 3
    assert len(result["D_energies"]) == 3
    assert result["D12_error_vs_direct_ci"] < 2e-4
