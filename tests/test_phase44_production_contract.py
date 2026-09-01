import sys
import subprocess

import torch

import scripts.benchmark_phase44_n4_explicit_correlation_d_gate as phase44


def test_phase44_import_does_not_cross_reference_firewall() -> None:
    command = (
        "import sys; "
        "import scripts.benchmark_phase44_n4_explicit_correlation_d_gate; "
        "assert 'scripts.phase44_reference_comparators' not in sys.modules; "
        "assert 'phase44_reference_comparators' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_phase44_frozen_initialization_and_configs() -> None:
    _, source = phase44._load_reference_free_fixture(phase44.DEFAULT_FIXTURE)
    for basis_order in phase44.D_AXIS:
        carrier = phase44._initial_carrier(source, basis_order)
        assert carrier.shape == (basis_order, 4)
        torch.testing.assert_close(
            carrier.mT @ carrier,
            torch.eye(4, dtype=torch.float64),
            atol=2e-15,
            rtol=2e-15,
        )
    optimizer = phase44._optimizer_config(44041)
    assert optimizer.steps == 100
    assert optimizer.chains == 32
    assert optimizer.samples_per_chain == 128
    assert optimizer.learning_rate == 0.01
    assert optimizer.final_learning_rate == 0.001
    selection = phase44._selection_config(45041)
    assert selection.samples_per_chain == 2000
    assert selection.chains == 32
    confirmation = phase44._confirmation_config(44241)
    assert confirmation.samples_per_chain == 5000
    assert confirmation.chains == 64


def test_phase44_ledger_selection_uses_energy_without_reference(tmp_path) -> None:
    optimizer_records = [
        {"D": basis_order, "lineage": lineage, "seed": 100 + lineage, "state_sha256": f"{basis_order}-{lineage}"}
        for basis_order in phase44.D_AXIS
        for lineage in (1, 2)
    ]
    selection_records = [
        {
            "D": basis_order,
            "lineage": lineage,
            "optimizer_seed": 100 + lineage,
            "seed": 200 + lineage,
            "energy": float(basis_order) + (0.1 if lineage == 1 else 0.2),
            "energy_standard_error": 0.01,
            "state_sha256": f"{basis_order}-{lineage}",
        }
        for basis_order in phase44.D_AXIS
        for lineage in (1, 2)
    ]
    fixture = tmp_path / "fixture.json"
    fixture.write_text("{}\n", encoding="utf-8")
    comparator_modules_before = {
        name: name in sys.modules
        for name in (
            "scripts.phase44_reference_comparators",
            "phase44_reference_comparators",
        )
    }
    ledger = phase44._selection_ledger(
        fixture, optimizer_records, selection_records
    )
    assert [choice["selected_lineage"] for choice in ledger["choices"]] == [1, 1, 1]
    assert comparator_modules_before == {
        name: name in sys.modules for name in comparator_modules_before
    }
