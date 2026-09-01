import json
from pathlib import Path

from femps.algorithms import load_slater_source_command_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "docs" / "experiments" / "configs" / "phase39_n4_d8_k4.json"


def test_restored_phase39_n4_d8_configuration_is_frozen() -> None:
    config, record = load_slater_source_command_config(CONFIG)
    assert (config.particles, config.basis_order, config.max_terms) == (4, 8, 4)
    assert config.source_optimizer.seed == 4001
    assert [
        (stage.target_terms, stage.candidate_seed, stage.optimizer_seed)
        for stage in config.stages
    ] == [
        (2, 4011, 4012),
        (3, 4021, 4022),
        (4, 4031, 4032),
    ]
    assert record["evidence_level"] == "numerical"
    assert record["acceptance"] == {
        "resume_energy_tolerance": 1e-11,
        "energy_nesting_tolerance": 1e-9,
        "source_ci_error_maximum": 0.002,
        "source_variance_maximum": 0.01,
        "final_ci_error_maximum": 1e-6,
        "final_variance_maximum": 1e-5,
        "optimizer_failure_count_maximum": 0,
        "norm_error_maximum": 1e-10,
        "antisymmetry_residual_maximum": 1e-12,
        "factorization_error_maximum": 1e-11,
        "stage_wall_time_maximum_seconds": 180.0,
        "command_wall_time_maximum_seconds": 900.0,
        "peak_cpu_rss_maximum_bytes": 2147483648,
    }


def test_restored_phase39_documents_internal_only_boundary() -> None:
    active = (
        ROOT / "docs" / "exec-plans" / "completed" / "phase39_n4_d8_addendum.md"
    ).read_text(encoding="utf-8")
    adr = (
        ROOT / "docs" / "decisions" / "0031-preregister-restored-phase39-n4-d8.md"
    ).read_text(encoding="utf-8")
    assert "internal **numerical evidence**" in active
    assert "No additional small numerical point" in active
    assert "outcome is retained" in adr
    assert "method-paper claim" in adr


def test_restored_phase39_config_is_valid_json_without_result_dependency() -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert record["checkpoint_path"].endswith("resumed.pt")
    assert record["output_path"].endswith("phase39_n4_d8_clean_source.json")
