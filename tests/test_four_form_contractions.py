import importlib.util
from fractions import Fraction
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "math" / "four_forms" / "exact_contractions.py"
SPEC = importlib.util.spec_from_file_location("four_form_exact", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ff)

SEARCH_MODULE_PATH = ROOT / "math" / "four_forms" / "explore_hypergraph_forms.py"
SEARCH_SPEC = importlib.util.spec_from_file_location(
    "four_form_hypergraph_search", SEARCH_MODULE_PATH
)
assert SEARCH_SPEC is not None and SEARCH_SPEC.loader is not None
search_module = importlib.util.module_from_spec(SEARCH_SPEC)
SEARCH_SPEC.loader.exec_module(search_module)


def test_four_dimensional_volume_has_expected_exact_ranks() -> None:
    form = ff.volume_form((0, 1, 2, 3))

    assert ff.four_form_hilbert_vector(form, 4) == (1, 4, 6, 4, 1)
    assert ff.is_concise(form, 4)


def test_five_dimensional_four_form_has_a_contraction_radical() -> None:
    form = ff.canonical_form(
        {
            (1, 2, 3, 4): 1,
            (0, 2, 3, 4): 2,
            (0, 1, 3, 4): 3,
            (0, 1, 2, 4): 5,
            (0, 1, 2, 3): 7,
        }
    )

    assert ff.contraction_rank(form, 5, 1) == 4
    assert not ff.is_concise(form, 5)


def test_six_dimensional_symplectic_dual_has_full_middle_rank() -> None:
    symplectic_form = ff.canonical_form(
        {(0, 1): 1, (2, 3): 1, (4, 5): 1}
    )
    four_form = ff.hodge_dual(symplectic_form, 6)

    assert ff.four_form_hilbert_vector(four_form, 6) == (1, 6, 15, 6, 1)
    assert ff.is_concise(four_form, 6)


def test_middle_contraction_is_symmetric_and_one_three_are_signed_transposes() -> None:
    form = ff.canonical_form(
        {
            (0, 1, 2, 3): 2,
            (0, 1, 4, 5): -3,
            (0, 2, 4, 6): Fraction(5, 7),
            (1, 3, 5, 6): 11,
        }
    )
    c1 = ff.contraction_matrix(form, 7, 1)
    c2 = ff.contraction_matrix(form, 7, 2)
    c3 = ff.contraction_matrix(form, 7, 3)

    assert c2 == [list(row) for row in zip(*c2, strict=True)]
    assert c3 == [[-c1[column][row] for column in range(len(c1))] for row in range(len(c1[0]))]


def test_direct_sum_adds_support_and_middle_ranks() -> None:
    volume = ff.volume_form((0, 1, 2, 3))
    form, dimension = ff.direct_sum(*[(volume, 4) for _ in range(4)])

    assert dimension == 16
    assert ff.four_form_hilbert_vector(form, dimension) == (1, 16, 24, 16, 1)
    assert ff.is_concise(form, dimension)


def test_basis_permutation_preserves_all_contraction_ranks() -> None:
    form = ff.canonical_form(
        {
            (0, 1, 2, 3): 1,
            (0, 1, 4, 5): 2,
            (0, 2, 4, 6): -1,
            (1, 3, 5, 7): 3,
            (2, 4, 6, 7): 5,
        }
    )
    permutation = (7, 3, 5, 1, 6, 0, 4, 2)

    transformed = ff.permute_basis(form, permutation)

    assert ff.four_form_hilbert_vector(transformed, 8) == ff.four_form_hilbert_vector(form, 8)


def test_hodge_dual_preserves_four_form_contraction_ranks_in_dimension_eight() -> None:
    form = ff.canonical_form(
        {
            (0, 1, 2, 3): 1,
            (0, 1, 4, 5): -2,
            (0, 2, 6, 7): 3,
            (1, 3, 5, 7): 5,
            (2, 4, 5, 6): -7,
        }
    )

    dual = ff.hodge_dual(form, 8)

    assert ff.hodge_dual(dual, 8) == form
    assert ff.four_form_hilbert_vector(dual, 8) == ff.four_form_hilbert_vector(form, 8)


def test_integer_ranks_agree_over_q_and_safe_recorded_primes() -> None:
    form, dimension = ff.direct_sum(
        (ff.volume_form((0, 1, 2, 3), 2), 4),
        (ff.volume_form((0, 1, 2, 3), 3), 4),
    )
    matrix = ff.contraction_matrix(form, dimension, 2)
    integer_matrix = [[int(value) for value in row] for row in matrix]

    assert ff.rational_rank(matrix) == 12
    assert ff.rank_mod_prime(integer_matrix, 5) == 12
    assert ff.rank_mod_prime(integer_matrix, 7) == 12


def test_rejects_float_coefficients_and_noncanonical_coordinates() -> None:
    with pytest.raises(TypeError, match="int or Fraction"):
        ff.canonical_form({(0, 1, 2, 3): 1.0})
    with pytest.raises(ValueError, match="increasing"):
        ff.contraction_matrix({(1, 0, 2, 3): 1}, 4, 2)


def test_hypergraph_screen_is_seeded_and_explicitly_nonproof() -> None:
    first = search_module.search(
        ambient_dimension=8, term_count=3, samples=20, seed=270008
    )
    second = search_module.search(
        ambient_dimension=8, term_count=3, samples=20, seed=270008
    )

    assert first == second
    assert first["evidence_status"] == "numerical evidence"
    assert first["screen_field"] == "F_2"
