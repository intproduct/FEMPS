# Phase 20 report: bounded coefficient algebras collapse to LC-AGP

## Decision

Candidate class H2 is closed negatively. A finite-dimensional complex
matrix-pair coefficient algebra does not define a distinct exactly tractable
FEMPS family when both its largest semisimple matrix block and its Jacobson-
radical nilpotency index are uniformly bounded. Every arbitrary-boundary
N=2M state then has an exact polynomial-size LC-AGP expansion.

This closes Gate H as a negative classification for the predeclared bounded-
algebra candidate class. It is not an affirmative method or novelty claim.

## Bound

For

```text
A/R = direct_sum_(a=1)^q Mat_(p_a)(C),
p = max_a p_a,   rho = dim R,   R^d = 0,
```

the constructive term bound is

```text
K <= sum_(k=0)^min(d-1,M)
       q^(k+1) rho^k binom(M+(k+1)p^2+k-1,(k+1)p^2+k-1).
```

At fixed `p,d`, this is polynomial jointly in particle-pair count, physical
basis dimension, and coefficient-algebra/virtual dimension. Existing finite-
AGP transition contractions therefore supply the norm and factorized one-/two-
body contractions, but only by reducing the state to an established LC-AGP
organization.

## Exact base-case evidence

- `T_2`, with square-zero radical, collapses to at most
  `binom(M+2,2)+2` AGPs. The exact M=1--6 certificate hash is
  `f671c2c10376c39cfb8c223edafba370570b9c417e8b12bcaaeb2f0f66cf078c`.
- The fully noncommutative semisimple algebra `Mat_2` collapses to at most
  `binom(M+3,3)` AGPs. A generic symbolic matrix with deterministic nonzero
  rational boundaries is reconstructed exactly for M=1--4; the certificate
  hash is
  `74d2de4a2cedcbaf548cd4c9895d0ea4af48e6bb485b72966562e46277a6d20d`.

The certificates validate independent exact constructions at the two smallest
noncommutative endpoints. The general theorem rests on the Wedderburn--radical
word expansion and polarization proof and remains labeled a theorem draft
pending external review.

## Research implication

Noncommutativity alone is not the missing fermionic correlation resource.
Within this coefficient-algebra design space, escaping polynomial LC-AGP
requires a simple block size or radical memory depth that grows with the
problem, while Phase 13 shows that sufficiently expressive growing memory can
recover a hard row-ordered determinant. The remaining research problem is
therefore narrower: identify additional structure on a growing family that
still permits exact polynomial contraction, or move to a controlled approximate
contraction with an explicit error certificate.

## Reproduction

```powershell
.\.venv\Scripts\python math\certificates\verify_triangular_pair_collapse.py --verify math\certificates\triangular_pair_lc_agp_certificate.json
.\.venv\Scripts\python math\certificates\verify_mat2_pair_collapse.py --verify math\certificates\mat2_pair_lc_agp_certificate.json
.\.venv\Scripts\python -m pytest -q tests\test_triangular_pair.py tests\test_exact_certificates.py
```
