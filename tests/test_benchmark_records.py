import pytest

from femps.benchmarks import (
    direct_exterior_feasibility,
    soft_coulomb_point_from_training,
)


def test_direct_exterior_budget_matches_combinatorial_sector() -> None:
    d14_n4 = direct_exterior_feasibility(4, 14)
    assert d14_n4.exterior_dimension == 1001
    assert d14_n4.dense_complex128_bytes == 16 * 1001**2
    assert d14_n4.feasible
    assert not direct_exterior_feasibility(4, 16).feasible


def test_soft_coulomb_record_separates_error_axes() -> None:
    training = {
        "config": {
            "interaction_model": "soft_coulomb",
            "particles": 4,
            "basis_order": 10,
            "agp_terms": 5,
            "soft_coulomb_quadrature_order": 128,
            "seed": 301,
        },
        "initialization": "provided_pair_matrices",
        "final_energy": 11.1,
        "finite_basis_reference_energy": 11.0,
        "finite_basis_ground_fidelity": 0.99,
        "polynomial_explicit_absolute_difference": 2e-13,
        "antisymmetry_residual": 0.0,
        "elapsed_seconds_this_call": 2.5,
        "peak_cuda_memory_bytes": 4096,
        "environment": {"device": "cuda:2"},
    }
    conditioning = {
        "retained_rank": 5,
        "discarded_rank": 0,
        "balanced_overlap_condition_number": 2.0,
        "raw_overlap_condition_number": 3.0,
        "pruning_or_restart_events": [],
    }
    point = soft_coulomb_point_from_training(
        "n4-d10-k5-s301",
        training,
        conditioning,
        largest_basis_reference_energy=10.95,
        direct_dense_same_basis_reference_energy=10.98,
    )
    assert point.correlation_optimizer_error == pytest.approx(0.1)
    assert point.operator_error_estimate == pytest.approx(0.02)
    assert point.basis_error_estimate == pytest.approx(0.03)
    assert point.total_error_estimate == pytest.approx(0.15)
    assert point.to_dict()["pruning_or_restart_events"] == []


def test_soft_coulomb_record_rejects_non_soft_coulomb_artifact() -> None:
    with pytest.raises(ValueError, match="not soft Coulomb"):
        soft_coulomb_point_from_training(
            "bad", {"config": {"interaction_model": "harmonic"}}, {}
        )
