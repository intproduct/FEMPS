"""Optimization algorithms admitted after the contraction gate."""

from .agp_subspace import (
    GeneralizedEigenResult,
    OverlapWhitening,
    TermPruningAssessment,
    assess_term_pruning,
    contribution_gram_spectrum,
    leave_one_out_energies,
    overlap_whitening,
    solve_generalized_hermitian,
)
from .finite_agp_training import (
    FiniteAgpConfig,
    canonical_pair_matrices,
    run_finite_agp_variable_projection,
)
from .pfaffian_training import (
    FactorizedPairConfig,
    PfaffianPairConfig,
    run_factorized_pfaffian_pair,
    run_pfaffian_pair,
)
from .diagonal_path_training import (
    DiagonalPathConfig,
    canonical_slater_orbitals,
    embed_diagonal_path_orbitals,
    extend_diagonal_path_terms,
    run_diagonal_path_variable_projection,
)
from .diagonal_path_contract import (
    DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION,
    DIAGONAL_PATH_RESULT_SCHEMA_VERSION,
    load_diagonal_path_checkpoint,
    validate_diagonal_path_checkpoint,
    validate_diagonal_path_result,
)

__all__ = [
    "GeneralizedEigenResult",
    "OverlapWhitening",
    "TermPruningAssessment",
    "assess_term_pruning",
    "contribution_gram_spectrum",
    "leave_one_out_energies",
    "overlap_whitening",
    "solve_generalized_hermitian",
    "FiniteAgpConfig",
    "canonical_pair_matrices",
    "run_finite_agp_variable_projection",
    "FactorizedPairConfig",
    "PfaffianPairConfig",
    "run_factorized_pfaffian_pair",
    "run_pfaffian_pair",
    "DiagonalPathConfig",
    "canonical_slater_orbitals",
    "embed_diagonal_path_orbitals",
    "extend_diagonal_path_terms",
    "run_diagonal_path_variable_projection",
    "DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION",
    "DIAGONAL_PATH_RESULT_SCHEMA_VERSION",
    "load_diagonal_path_checkpoint",
    "validate_diagonal_path_checkpoint",
    "validate_diagonal_path_result",
]
