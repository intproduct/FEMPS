# Mathematical research code

This tree is reserved for exact/certificate-oriented work on antisymmetric TT
ranks, exterior contractions, and four-forms. It is intentionally separate
from `src/femps`, which contains numerical solver code.

Floating-point exploration may guide conjectures but cannot satisfy a proof
test. Every certificate pipeline must record the base field, exact arithmetic
backend, parameters, seed where relevant, artifact hash, and independent
verification command.

`femps_no_go_manuscript.tex` is the third-draft structural/no-go manuscript. It
assembles the ordinary particle-TT rank results, fixed-bond and sparse exterior
contraction obstructions, restricted-algebra collapses, approximation boundary,
statistics-carrier obstruction, explicit AGP embedding, and the restricted
interacting FEMPS control in one submission-oriented source. Visible internal
claim/evidence labels are removed; floating calculations remain identified as
numerical evidence.

Build that sole submission source from the repository root with
`python scripts/build_combined_manuscript.py`. The driver invokes `pdflatex`
and `bibtex` directly, so Windows builds do not require the Perl dependency of
`latexmk`, and it rejects unresolved references or layout warnings in the final
log.

`generic_femps_contraction_obstruction.tex` now begins with the direct
site-indexed one-form Cayley identity and the conditional #P-hardness theorem
for exact squared norms at maximum bond three. It retains the Phase 13/14
shift-tagged homogeneous pair-power reduction as an independent mechanism.

`certificates/verify_fixed_bond_cayley.py` is an implementation-independent
exact-integer verifier for the direct theorem. For orders two through six it
checks all four boundary entries, explicit one-form virtual paths, the
bond-three scalar-reference polarization, and exact antisymmetry. Its committed
certificate hash is
`1d2208d3e5cb14f5c8e6c875f7fddf51c47ce9a3e61be6cedb8246d662b3a016`.

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

`certificates/verify_approximate_exterior_gate.py` supplies the exact Phase 24
conditioning controls for positive, cancelling, signed-PSD, and Rayleigh-
quotient cases. The approximation-complexity conclusion itself depends on the
published real-PSD permanent theorem, not on bounded enumeration.

`statistics_carrier_obstruction.tex` contains the Phase 25 cut-rank
divisibility obstruction to a universal direct statistics-carrier tensor
factorization. `certificates/verify_statistics_carrier_obstruction.py` checks
all cuts for `3<=N<=8`, component-channel locking, orbital permutations,
direct orbital embeddings, and full-support perturbations over exact
rationals.

`docs/theory/exterior_no_go_hierarchy.md` is the Phase 23 synthesis entry point.
It distinguishes ordinary particle-TT representation rank from exact exterior
contraction complexity and records the coverage boundary of every theorem.
