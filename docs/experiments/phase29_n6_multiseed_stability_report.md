# Phase 29 N=6 multiseed stability report

## Scope

ADR 0019 fixes three genuinely blind `N=6,D=10,K=4` soft-Coulomb starts at
seeds 31, 37, and 43. All use the same 160 Adam steps, 80 L-BFGS steps,
physical-operator SVD, and direct `Q=128` exterior-CI reference. No CI state,
K1 checkpoint, or seed-specific tuning is used.

This is **numerical stability evidence at one `(N,D,K)` point**, not a particle-
number, basis, determinant-count, or asymptotic scaling result. Reproduce and
verify with

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_phase29_n6_multiseed_stability.py
.\.venv\Scripts\python.exe scripts\verify_phase29_n6_multiseed_stability.py
```

Raw data are in `results/phase29_n6_multiseed_stability.json`.

## Result

The fixed finite-basis CI energy is `25.049639839832263`.

| Seed | FEMPS energy | CI error | Variance | Condition number | Time | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 25.049889217607940 | `2.493778e-4` | `2.168588e-3` | 3.394 | 112.14 s | 787,750,912 bytes |
| 37 | 25.049933069070477 | `2.932292e-4` | `2.333679e-3` | 3.209 | 105.06 s | 828,588,032 bytes |
| 43 | 25.050042962908936 | `4.031231e-4` | `3.094192e-3` | 4.424 | 105.92 s | 882,962,432 bytes |

All three errors are below the preregistered `5e-4` limit and all variances are
below `5e-3`. Energy spread is `1.537453e-4`, below the `2.5e-4` stability
limit. Every run retains all four overlap directions, has zero norm error and
zero structural antisymmetry residual, and enumerates zero virtual paths.

Seed 31 performs the million-coefficient validation materialization. Its
antisymmetry residual is zero and its particle-TT ranks are
`(10,45,80,45,10)`, versus `(10,45,120,45,10)` for dense CI. The other seeds
explicitly report a null materialized residual because their registered cap
disables that exponential validation; this is not a missing structural
residual.

The physical two-body factorization rank is 19 with dense relative error
`1.340e-15`. The maximum measured time is 112.14 s and maximum sampled RSS is
882,962,432 bytes, well below the 600 s and 1.5 GiB caps.

## Interpretation and stop decision

The independent verifier returns `multiseed_pass: true`. The N6 point is now
reproducible under blind fixed-budget optimization rather than only feasible
under one K1-continuation lineage. Initialization still matters quantitatively:
the earlier K1-to-K4 continuation reached error `1.382e-4`, below all three
blind values. That is an optimization tradeoff, not hidden by choosing only the
best lineage.

Particle-number expansion stops here. The present evidence does not justify an
N8 diagonal-path run because it would weaken direct truth/materialization while
changing the principal cost axis. The next task is a matched N4-to-N6
structural and contraction-cost audit at fixed `D,K,L`, followed by method
consolidation. Any future N8 proposal requires a new ADR with an independently
controlled reference and a physics question not answered at N6.
