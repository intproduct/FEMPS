from pathlib import Path

import torch

from femps.algorithms import (
    CorrelatedExteriorVMCOptimizerConfig,
    canonical_lowest_orbitals,
    run_correlated_exterior_vmc_optimization,
)


def _config() -> CorrelatedExteriorVMCOptimizerConfig:
    return CorrelatedExteriorVMCOptimizerConfig(
        particles=2,
        chains=4,
        steps=4,
        burn_in_sweeps=4,
        rethermalization_sweeps=2,
        samples_per_chain=8,
        thinning_sweeps=2,
        proposal_scale=0.7,
        seed=43101,
        learning_rate=0.01,
        final_learning_rate=0.002,
        gradient_clip_norm=1.0,
        amplitude_bound=1.0,
        checkpoint_every=1,
        max_autocorrelation_lag=4,
        coupling=1.0,
        softening=1.0,
    )


def test_stochastic_optimizer_resume_matches_clean_trajectory(tmp_path: Path) -> None:
    config = _config()
    raw = canonical_lowest_orbitals(3, 2)
    amplitudes = torch.tensor([-0.05], dtype=torch.float64)
    exponents = torch.tensor([1.0], dtype=torch.float64)
    checkpoint = tmp_path / "optimizer.pt"
    partial = run_correlated_exterior_vmc_optimization(
        config,
        raw,
        amplitudes,
        exponents,
        checkpoint_path=checkpoint,
        max_steps_this_call=2,
    )
    assert partial["completed"] is False
    assert partial["completed_steps"] == 2
    resumed = run_correlated_exterior_vmc_optimization(
        config,
        raw,
        amplitudes,
        exponents,
        checkpoint_path=checkpoint,
        resume=True,
    )
    clean = run_correlated_exterior_vmc_optimization(
        config, raw, amplitudes, exponents
    )
    assert resumed["completed"] is True
    assert resumed["completed_steps"] == clean["completed_steps"] == 4
    assert resumed["history"] == clean["history"]
    assert resumed["accepted_proposals"] == clean["accepted_proposals"]
    assert resumed["total_proposals"] == clean["total_proposals"]
    assert resumed["acceptance_rate"] == clean["acceptance_rate"]
    assert torch.equal(resumed["raw_orbitals"], clean["raw_orbitals"])
    assert torch.equal(resumed["orbitals"], clean["orbitals"])
    assert torch.equal(resumed["amplitudes"], clean["amplitudes"])
    assert resumed["materialization"] == {
        "D_to_the_N_tensor": False,
        "full_alternating_coefficient_tensor": False,
        "virtual_paths": 0,
    }


def test_stochastic_optimizer_rejects_changed_resume_identity(tmp_path: Path) -> None:
    config = _config()
    raw = canonical_lowest_orbitals(3, 2)
    amplitudes = torch.tensor([-0.05], dtype=torch.float64)
    exponents = torch.tensor([1.0], dtype=torch.float64)
    checkpoint = tmp_path / "optimizer.pt"
    run_correlated_exterior_vmc_optimization(
        config,
        raw,
        amplitudes,
        exponents,
        checkpoint_path=checkpoint,
        max_steps_this_call=1,
    )
    changed = amplitudes.clone()
    changed[0] += 0.01
    try:
        run_correlated_exterior_vmc_optimization(
            config,
            raw,
            changed,
            exponents,
            checkpoint_path=checkpoint,
            resume=True,
        )
    except ValueError as error:
        assert "initialization mismatch" in str(error)
    else:
        raise AssertionError("changed optimizer initialization was accepted")
