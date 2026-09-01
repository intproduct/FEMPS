from __future__ import annotations

import copy
import json
from pathlib import Path

from femps.algorithms import load_slater_source_command_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_A = ROOT / "docs/experiments/configs/phase38_n4_d6_k4_seed_a.json"
CONFIG_B = ROOT / "docs/experiments/configs/phase38_n4_d6_k4_seed_b.json"


def _without_registered_seeds_and_paths(record: dict) -> dict:
    value = copy.deepcopy(record)
    value["source_optimizer"]["seed"] = None
    for stage in value["adaptive"]["stages"]:
        stage["candidate_seed"] = None
        stage["optimizer_seed"] = None
    value["checkpoint_path"] = None
    value["output_path"] = None
    return value


def test_phase38_configs_change_only_registered_seeds_and_paths() -> None:
    config_a, record_a = load_slater_source_command_config(CONFIG_A)
    config_b, record_b = load_slater_source_command_config(CONFIG_B)
    assert _without_registered_seeds_and_paths(record_a) == (
        _without_registered_seeds_and_paths(record_b)
    )
    assert config_a.source_optimizer.seed == 3801
    assert [(stage.candidate_seed, stage.optimizer_seed) for stage in config_a.stages] == [
        (3811, 3812),
        (3821, 3822),
        (3831, 3832),
    ]
    assert config_b.source_optimizer.seed == 3901
    assert [(stage.candidate_seed, stage.optimizer_seed) for stage in config_b.stages] == [
        (3911, 3912),
        (3921, 3922),
        (3931, 3932),
    ]


def test_phase38_preregistered_acceptance_is_identical_and_bounded() -> None:
    record_a = json.loads(CONFIG_A.read_text(encoding="utf-8"))
    record_b = json.loads(CONFIG_B.read_text(encoding="utf-8"))
    assert record_a["acceptance"] == record_b["acceptance"]
    acceptance = record_a["acceptance"]
    assert acceptance["final_ci_error_maximum"] == 1e-6
    assert acceptance["final_variance_maximum"] == 1e-5
    assert acceptance["final_energy_spread_maximum"] == 2e-6
    assert acceptance["fresh_difference_from_phase37_maximum"] == 1e-6
    assert acceptance["optimizer_failure_count_maximum"] == 0
