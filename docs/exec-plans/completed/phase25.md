# Completed execution plan: Phase 25 statistics-carrier/multiplicity gate

## Objective

Test whether exchange statistics can be isolated in a fixed structural carrier
while a canonical multiplicity space stores only correlations beyond a Slater.

## Completed checkpoints

- [x] Defined the factored object as the image of the exterior contraction map
  at each particle cut.
- [x] Audited symmetry-adapted structural/degeneracy tensors, non-Abelian
  QSpace, sign and exterior representations, Slater rank, and Grassmannian
  secant identifiability.
- [x] Enforced the Slater multiplicity-one condition without selecting a hidden
  occupied-orbital gauge.
- [x] Proved a two-Slater rank-divisibility counterexample for every `N>=3`.
- [x] Checked all cuts, component-channel locking, orbital permutations,
  direct embeddings, and full-support perturbations exactly for `3<=N<=8`.
- [x] Distinguished Hamiltonian-specific symmetry sectors and state-adaptive
  Slater sums from the rejected universal direct tensor product.
- [x] Showed that ordinary Schmidt singular values remain the only supplied
  canonical truncation spectrum and retain the Slater flat-spectrum cost.
- [x] Reapplied the Phase 22 permanent path to show that projective Slater
  multiplicity one does not imply compact-input norm/reconstruction.
- [x] Issued Gate L and ADR 0015 before optimizer/GPU work.

## Result

Gate L is **FAIL** for Candidate L1. A fixed carrier whose one-cut dimension is
forced to `N` cannot tensor-factor a valid state of cut rank `N+2`. Natural
symmetry decompositions either provide only the one-dimensional sign irrep or
retain the full orbital exterior representation. State-adaptive determinant
channels are established nonorthogonal Slater/AGP expansions rather than a
canonical free multiplicity space.

The decision does not reject all categorical structures or physical symmetries.
Any successor must define a different object and prove exactness, safe
truncation, and compact-input contraction from scratch.
