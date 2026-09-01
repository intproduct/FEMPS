# Phase 34 truth-free adaptive N6 correlation-growth report

## Scope and preregistration

ADR 0023 froze one interacting `N=6,D=12` soft-Coulomb continuation before the
production result was inspected. Starting from the accepted Phase 32 K4 state,
each growth step generated exactly 32 seeded Slater candidates. Candidates
were ranked only by factorized determinant-transition energies and balanced
overlap conditioning. Dense CI data were constructed only after the adaptive
K5, adaptive K6, and same-budget cold K6 optimizations had all finished.

This is numerical evidence for one restricted first-quantized continuous FEMPS
point. It is not an N-scaling, generic-contraction, or superiority result.

## Results

| State | Energy | Error versus CI | Variance | Condition | Time (s) | Peak RSS (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| Phase 32 source K4 | 25.049471144618 | 1.04729e-4 | 1.12233e-3 | 2.111 | 107.66 | 974,974,976 |
| Adaptive K5 | 25.049434533568 | 6.81175e-5 | 7.74670e-4 | 2.440 | 6.72 | 656,408,576 |
| Adaptive K6 | 25.049398437461 | 3.20214e-5 | 3.72621e-4 | 2.760 | 6.24 | 663,953,408 |
| Cold K6 | 25.049642546632 | 2.76131e-4 | 2.65947e-3 | 8.039 | 6.28 | 667,652,096 |

The adaptive K6 state lowers the K4 energy by `7.27072e-5` and the variance by
`7.49710e-4`. It is `2.44109e-4` below the matched cold K6 state. The selected
pool indices are 31 and 23. Their fixed-span predicted improvements are only
`3.59564e-7` and `3.27237e-7`; nonlinear reoptimization supplies the larger
final gains.

All optimized points retain every K direction, have norm errors below
`1.6e-15`, zero structural antisymmetry residual, and zero production virtual-
path or particle-tensor enumeration. The K6 gauge-balanced pruning assessment
does not trigger. Its ordinary particle-TT ranks are
`(12,66,120,66,12)`: 209,376 TT scalars versus 438 stored FEMPS orbital and
amplitude scalars. This parameter comparison is structural and does not imply
runtime superiority; direct CI remains exact and faster in this truth space.

The source K4 time was recorded before Phase 33 vectorization and is not a
matched timing comparator for K5/K6. Phase 34 establishes an algorithmic growth
capability and a cold-start advantage for this registered seed pair. It does
not yet establish adaptive-pool multiseed stability.

## Verification

Primary artifact:
`docs/experiments/results/phase34_adaptive_k_growth.json`.

Independent verifier:
`scripts/verify_phase34_adaptive_k_growth.py`.
