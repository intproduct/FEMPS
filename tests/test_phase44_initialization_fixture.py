from pathlib import Path

from scripts.verify_phase44_initialization_fixture import verify


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "docs"
    / "experiments"
    / "results"
    / "phase44_phase37_k1_initialization.json"
)


def test_phase44_initialization_fixture_is_exact_and_reference_free() -> None:
    result = verify(FIXTURE)
    assert result["verified"] is True
    assert result["source_terms"] == 1
    assert result["source_seed"] == 3701
    assert result["carrier_shape"] == [6, 4]
    assert result["maximum_gram_residual"] <= 2e-15
    assert result["reference_fields_present"] is False
