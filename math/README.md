# Mathematical research code

This tree is reserved for exact/certificate-oriented work on antisymmetric TT
ranks, exterior contractions, and four-forms. It is intentionally separate
from `src/femps`, which contains numerical solver code.

Floating-point exploration may guide conjectures but cannot satisfy a proof
test. Every certificate pipeline must record the base field, exact arithmetic
backend, parameters, seed where relevant, artifact hash, and independent
verification command.

`generic_femps_contraction_obstruction.tex` contains the Phase 13/14 tagged
Cayley-determinant reduction and the conditional #P-hardness theorem for exact
generic matrix-wedge FEMPS norms.

`certificates/verify_tagged_cayley.py` is an implementation-independent exact
integer verifier. It enumerates perfect matchings and factor permutations
directly and checks `certificates/tagged_cayley_certificate.json` for orders
one through four. It imports neither PyTorch nor `femps`.

`certificates/verify_triangular_pair_collapse.py` independently verifies the
Phase 20 `T_2` upper-triangular matrix-pair collapse using exact rational
polynomial arithmetic. It checks both the symbolic matrix power and a rational
LC-AGP power interpolation for pair orders one through six, importing neither
PyTorch nor `femps`.

`certificates/verify_mat2_pair_collapse.py` checks the complementary semisimple
base case. It raises a generic symbolic `2 x 2` matrix over four commuting
two-form variables, applies deterministic nonzero rational boundaries, and
reconstructs the resulting pair powers by exact rational simplex interpolation.
The committed certificate covers M=1 through 4 and imports neither PyTorch nor
`femps`; arbitrary-boundary coverage follows separately from homogeneous-power
spanning.

`certificates/verify_truncated_polynomial_pair_collapse.py` checks the Phase 21
growing-radical candidate `Q[z]/(z^d)`. For every `1<=M,d<=4`, it constructs
exact rational coefficient-extraction weights, expands both sides in all pair-
form coordinates, and verifies every boundary basis functional. This proves
arbitrary-boundary coverage by linearity for the certified cases.

`certificates/verify_alternating_word_pair_collapse.py` checks the minimal
noncommutative growing-memory algebra with `x^2=y^2=0`. It compares direct word-
algebra powers with the nested exact decomposition from its
`Mat_2(Q[z]/(z^d))` embedding for every boundary word, `1<=M<=3`, and
`1<=d<=4`.

`certificates/verify_sparse_path_apg_permanent.py` checks the Phase 22
upper-bidiagonal APG permanent reduction. It compares exact virtual-path,
square-zero exterior-subset, and permutation-enumeration routes for three
matrix families at every `1<=M<=6`; it also records the normalized squared norm
as an exact rational.

`docs/theory/exterior_no_go_hierarchy.md` is the Phase 23 synthesis entry point.
It distinguishes ordinary particle-TT representation rank from exact exterior
contraction complexity and records the coverage boundary of every theorem.
