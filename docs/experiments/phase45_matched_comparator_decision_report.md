# Phase 45 matched-comparator decision report

## Outcome

No new comparator calculation is authorized. Same-basis DMRG is redundant in
the current exact-CI region, while the existing Li--Waintal-style ordered-
coordinate result is not matched to Phase 44 in basis, domain, representation,
error control, or cost. Producing a table now would add method names without
answering a new scientific question.

## Same-basis DMRG audit

For four particles, the Phase 44 harmonic-orbital exterior dimensions are:

| `D` | `binom(D,4)` |
|---:|---:|
| 4 | 1 |
| 6 | 15 |
| 8 | 70 |

The repository already constructs and diagonalizes the direct four-index
Slater--Condon Hamiltonian in these spaces. It gives exact projected energies,
eigenvectors, variances, and independent quadrature controls. A
second-quantized DMRG calculation would approximate these known answers while
introducing orbital ordering, bond dimension, truncation, sweep, and software
choices. It cannot strengthen the current accuracy claim or test an NOCI-
inaccessible physical effect.

ADR 0020's admission conditions therefore remain unmet: direct CI is not
materially infeasible, and no new physics/complexity question has appeared.

## Li--Waintal/ordered-coordinate audit

The existing controlled N4 ordered-coordinate point has:

- center-of-mass plus three positive-gap coordinates;
- a finite half-line sine basis with local order 10 and `Rmax=4.5`;
- polynomial soft-Coulomb degree 20;
- MPS bond 32 and 6,600 parameters;
- total error `4.375e-3--4.393e-3` against the exterior D14 numerical
  reference, dominated by basis/box error;
- optimization error `2.84e-5--4.63e-5` against its own Galerkin truth.

Phase 44 uses full-line harmonic orbitals, carrier `D=4,6,8`, one exterior
Slater, five symmetric correlation features, and stochastic coordinate
estimation. At D8 it has 37 raw carrier/correlator parameters before gauge
quotients. The two uses of `D` do not describe the same one-particle space or
cutoff; the ordered state also distributes approximation across domain,
interaction polynomial, local basis, MPS bond, and optimizer axes that Phase
44 does not share.

A raw energy/time comparison would therefore conflate representation choice
with unmatched basis and operator errors. The ordered route must remain named
Li--Waintal/Hong-related work, not FEMPS.

## Decision and next handoff

ADR 0034 records the negative decision. No larger N/D, DMRG run, ordered-
coordinate rerun, or new FEMPS sample is launched. A future matched comparison
requires a new preregistration with a common accuracy target and complete error
and resource budgets.

The immediate deliverable is instead an external-review handoff:

1. algebraic-complexity review of the CHSS-based exact-norm theorem and the
   rational-Legendre pointwise reduction;
2. external clean reproduction of the Phase 44 failed gate and its low-D
   confirmation subresult; and
3. continued single-manuscript scope until either external evidence or a truly
   matched comparator passes.
