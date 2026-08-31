# Completed execution plan: Phase 3 contraction gate

## Objective

Determine whether a nontrivial `chi>1` FEMPS family admits exact contraction
with polynomial cost in `N`, `D`, and `chi`.

## Checkpoints

- [x] Cross-check the norm with full tensor, determinant-path, and exterior
  dynamic-programming implementations.
- [x] Cross-check one-body expectations with full tensor and Slater cofactors.
- [x] Cross-check a minimal two-body expectation with independent methods.
- [x] Verify AD gradients for every contraction implementation.
- [x] Measure time and exact structural-memory scaling for each exact route.
- [x] Derive explicit asymptotic costs and identify the exponential variable.
- [x] Investigate a polynomial generic recurrence or a physically meaningful
  polynomial subclass.
- [x] Issue a CONDITIONAL Gate A report admitting only the fixed-number
  Pfaffian/AGP structured subclass and finite sums.

## Exit criterion

Gate A is decided from explicit algorithms, cross-validation, complexity
evidence, and a documented fallback; parameter count alone is not evidence.
