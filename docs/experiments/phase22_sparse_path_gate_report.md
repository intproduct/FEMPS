# Phase 22 report: sparse path memory is APG-hard

## Decision

Gate J closes negatively. An upper-bidiagonal pair matrix with growing width
`w=M+1` and endpoint boundaries has one virtual path, but that path is exactly
an antisymmetrized product of geminals:

```text
e_0^T Omega^M e_M = F_1 wedge ... wedge F_M.
```

With paired orbitals `P_j` and `F_i=sum_j A_(i,j)P_j`, its only top-form
coefficient is `perm(A)`. Under the normalized pair-power convention,

```text
||Psi||^2 = perm(A)^2 / (M!)^2.
```

For 0--1 `A`, an exact squared-norm oracle therefore recovers a #P-complete permanent.
The construction has `D=2M`, width `M+1`, bandwidth one, one virtual path, and
only `O(M^2)` binary input coefficients. Tridiagonal and broader fixed-bandwidth
families inherit the obstruction because they contain this specialization.

## Exact evidence

The standalone certificate compares upper-bidiagonal path propagation,
square-zero pair-exterior propagation, and permutation enumeration. Identity,
all-one, and deterministic binary matrices pass for every `1<=M<=6` with exact
integer/rational arithmetic. Certificate hash:

```text
dd72c1aaeb0bc2a6b9206992cde9099f2f568b7ff6c8ed8eb7e38d958f78e790
```

## Corrected rank interpretation

The formal commutative monomial `F_1...F_M` has an exponential Fischer/Waring
decomposition, but its ordinary polynomial Waring rank is not automatically a
physical LC-AGP lower bound. Exterior multiplication has a large kernel; in the
hard reduction the final `2M`-form space is even one-dimensional. The result is
therefore a contraction-complexity obstruction, not a physical AGP-rank lower
bound.

## Prior-art boundary

The unique-path state is standard APG/APIG. APG-to-AGP Fischer decompositions,
permanent-valued determinant coefficients, and generic geminal contraction
difficulty are established in the geminal literature. The project contributes
only the sparse growing-width Gate J embedding, exact normalization certificate,
and its placement in the FEMPS no-go hierarchy.

## Consequence

No continuous solver, GPU optimization, or AD benchmark is admitted for this
candidate. Sparse amplitude propagation and polynomial parameter count do not
imply polynomial norm or observable contraction. The exterior exact-algebra
search has now exhausted bounded coefficient algebras, one-generator growing
memory, fixed-state graded memory, and generic fixed-bandwidth growing paths.

## Reproduction

```powershell
.\.venv\Scripts\python math\certificates\verify_sparse_path_apg_permanent.py --verify math\certificates\sparse_path_apg_permanent_certificate.json
.\.venv\Scripts\python -m pytest -q tests\test_exact_certificates.py
```
