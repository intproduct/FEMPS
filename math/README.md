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
