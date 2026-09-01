# Completed execution plan: Phase 34 Adaptive N6 Correlation Growth

## Objective

Turn the Phase 33 contraction speedup into a practical solver capability by
implementing and auditing truth-free adaptive growth of the diagonal-path
correlation multiplicity at the interacting `N=6,D=12` soft-Coulomb point.

## Completion record (2026-09-01)

- [x] ADR 0023 preregistered a 32-candidate fixed-span selector, growth seeds
  3451/3452, cold K6 seed 3460, frozen optimizer budgets, resource caps, and
  failure rules before production.
- [x] Candidate selection reads only factorized determinant-transition
  Hamiltonians, overlap matrices, generalized-eigenvalue energies, and balanced
  conditioning. Dense CI is constructed only after all optimizations freeze.
- [x] Small-system explicit exterior values agree within `1e-10`, reverse-mode
  gradients within `1e-8`, and materialized antisymmetry residual is below
  `1e-12`.
- [x] Adaptive energies decrease from `25.049471144618` at K4 to
  `25.049434533568` at K5 and `25.049398437461` at K6.
- [x] The same-basis CI error decreases from `1.04729e-4` to `3.20214e-5`; the
  variance decreases from `1.12233e-3` to `3.72621e-4`.
- [x] Adaptive K6 is `2.44109e-4` below the same-budget cold K6 control.
- [x] K6 retains all six directions with balanced condition `2.760`; the
  registered pruning assessment does not trigger.
- [x] Every production state has zero structural antisymmetry residual and zero
  virtual-path or particle-tensor enumeration. K5/K6/cold K6 each complete in
  under 6.8 CPU seconds and below 668 MB sampled peak RSS.
- [x] An independent verifier reconstructs the 924-dimensional CI reference,
  four energies, variances, norms, ordinary particle-TT ranks, storage counts,
  source hashes, selection arithmetic, and acceptance gates solely from the
  committed artifact.
- [x] The eleven-entry reproduction manifest, method manuscript, provenance,
  and visually inspected ten-page PDF include the bounded Phase 34 evidence.
- [x] The final repository suite passes: `264 passed`, with one pre-existing
  latticeTN scalar-reporting warning and no failures.

## Scientific decision

Phase 34 is **PASS as numerical algorithm evidence for one preregistered
adaptive lineage**. It demonstrates that a truth-free determinant-transition
selector can turn increased K into a measurable interacting-state improvement
and outperform a cold start at the same K and optimizer budget. It does not yet
establish candidate-pool multiseed stability, a universal stopping rule,
generic FEMPS efficiency, or runtime superiority over direct CI.

Primary record:
`docs/experiments/results/phase34_adaptive_k_growth.json`.

Independent verifier:
`scripts/verify_phase34_adaptive_k_growth.py`.
