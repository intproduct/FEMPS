# Completed execution plan: Phase 2 FEMPS algebra

## Objective

Define the matrix-wedge ansatz rigorously and validate its small-system
expressivity before attempting scalable contraction.

## Checkpoints

- [x] Fix field, normalization, boundary, and matrix-wedge conventions.
- [x] Prove associativity and strict antisymmetry.
- [x] Prove and test that `chi=1` is exactly the decomposable/Slater family.
- [x] Embed every finite weighted Slater sum into diagonal virtual paths.
- [x] Characterize `N=2`: minimal FEMPS bond equals half the skew-matrix rank.
- [x] Record the ordinary gauge action and unresolved canonical-form questions.
- [x] Cross-check matrix-wedge multiplication against independent virtual-path
  enumeration for `N=2,3,4` and small bonds.

## Exit criterion

A formal definition and explicit reference implementation agree for small
systems, with no contraction-complexity claim made before Gate A.
