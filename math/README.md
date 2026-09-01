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
