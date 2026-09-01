# Active execution plan: Phase 31 Restricted FEMPS Method Manuscript and Release

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
