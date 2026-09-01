# Completed execution plan: Phase 31 Restricted FEMPS Method Manuscript and Release

## Objective

Prepare a technically complete method-manuscript draft and reproducible
research release for the validated nonbranching diagonal-path FEMPS subclass.
The restricted scope must be visible at title/abstract level; generic FEMPS is
not claimed efficiently contractible.

## Required manuscript content

1. First-quantized continuous functional-basis definition and exact
   matrix-wedge embedding.
2. Nonbranching global path restriction and relation to a nonorthogonal
   K-Slater expansion.
3. Value/reverse-mode one- and two-body transition formulas, singular-safe
   fallback, physical operator-SVD, and explicit `N,D,K,L` complexity.
4. QR gauge, generalized Hermitian variable projection, AD, checkpoints,
   seeds, schemas, residual reporting, and failure rules.
5. N2/N4/N6 benchmark ladder with references, D/K convergence, multiseed
   stability, variance, norm, antisymmetry, time, memory, and named comparators.
6. Exchange-carrier versus correlation-multiplicity interpretation and measured
   N4/N6 rank/cost tradeoff.
7. Generic exact-contraction obstruction as an algorithm constraint.
8. Limitations: CI is faster in the current truth region; no N8, asymptotic,
   DMRG-superiority, generic-practicality, or categorical-novelty claim.

## Reproducible release gate

- Manifest and figure-provenance verifiers pass from a clean checkout.
- Every numerical table/figure maps to one manifest claim identifier.
- Ignored checkpoints have complete generation commands and are never treated
  as committed evidence.
- Public schemas remain frozen or receive an explicit migration note.
- Version may advance from `0.0.1` only after manuscript, manifest, tests, and
  documented install path pass together.

## Paper boundary

This method manuscript is separate from the structural/no-go paper. They may
cross-reference constraints, but numerical and exact mathematical claims retain
separate evidence labels.

## Completion record (2026-09-01)

- [x] Eight-artifact reproduction manifest, including an independent verifier
  for the admitted N2/N4 harmonic ladder.
- [x] Two provenance-hashed paper figures with automated source/output checks.
- [x] Complete restricted-method manuscript source and 8-page visually checked
  PDF with explicit scope, equations, complexity, solver contract, N2/N4/N6
  evidence, comparators, limitations, and reproduction mapping.
- [x] Manuscript evidence lint requiring all manifest identifiers and mappings
  for every numerical table and figure.
- [x] Source/manifest/figure/PDF SHA-256 provenance and a reproducible build
  script that does not depend on latexmk/Perl.
- [x] Full repository test suite: 252 passed; the sole warning is the known
  latticeTN report-path tensor-to-scalar warning.
- [x] Release-candidate commit prepared for push to `main`.
