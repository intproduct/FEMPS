from pathlib import Path

from scripts.verify_phase28_diagonal_path_ladder import verify_artifact


def test_recorded_diagonal_path_ladder_admitted_subset_passes() -> None:
    result = verify_artifact(
        Path("docs/experiments/results/phase28_diagonal_path_ladder.json")
    )
    assert result["verified"]
    assert result["admitted_points"] == 7
