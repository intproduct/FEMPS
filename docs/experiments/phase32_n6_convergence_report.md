# Phase 32 N=6 independent D/K convergence report

All values below are floating-point **numerical evidence** for the restricted
nonbranching diagonal-path FEMPS subclass. Direct CI is an independent
same-functional-basis reference and is never used to initialize FEMPS.

## Correlation axis at D=10

Every point uses 160 Adam and 80 LBFGS steps.

| K | Energy | Error vs CI | Variance | Time (s) | Peak RSS (bytes) | Ordinary particle-TT ranks | FEMPS scalars |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 25.052242782725 | 2.602943e-3 | 1.477540e-2 | 6.82 | 759472128 | (6,15,20,15,6) | 61 |
| 2 | 25.050276338618 | 6.364988e-4 | 4.695188e-3 | 25.62 | 780746752 | (10,29,40,29,10) | 122 |
| 4 | 25.049825287522 | 1.854477e-4 | 1.748982e-3 | 88.15 | 856158208 | (10,45,80,45,10) | 244 |

The direct D10 CI energy is `25.049639839832`. Energies and variances decrease
as K grows. This fixed-lineage convergence axis is supplemented, not replaced,
by Phase 29's blind K4 seeds 31, 37, and 43, all using the same 160/80 budget.

## Functional-basis axis at K=4

| D | FEMPS energy | Direct CI energy | Error vs CI | Variance | Time (s) | Peak RSS (bytes) | Ordinary particle-TT ranks |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 25.051264850894 | 25.051232892768 | 3.195813e-5 | 3.015810e-4 | 74.68 | 901804032 | (8,28,56,28,8) |
| 10 | 25.049825287522 | 25.049639839832 | 1.854477e-4 | 1.748982e-3 | 88.15 | 856158208 | (10,45,80,45,10) |
| 12 | 25.049471144618 | 25.049366416097 | 1.047285e-4 | 1.122331e-3 | 107.66 | 974974976 | (12,60,80,60,12) |

Both FEMPS and direct-CI absolute energies decrease with D. The per-basis FEMPS
error is not monotone: the larger functional space adds correlation directions
that fixed K must approximate. This is an interpretable D/K tradeoff, not a
continuum-error bound.

## Operator, symmetry, and representation audits

At D=8,10,12, the physical operator-SVD ranks are 15, 19, and 23. Relative
factorization errors are below `4e-15`; Q128-to-Q160 dense-operator changes are
`2.79e-13`, `6.35e-13`, and `1.35e-12`. All structural antisymmetry residuals
are exactly zero. D8 and D10 K4 materializations independently give zero
residual; D12 does not materialize `12^6` coefficients.

The D12 K4 state stores 292 complex FEMPS scalars. Its compactly reconstructed
ordinary particle TT needs ranks `(12,60,80,60,12)` and 132,768 scalars. The
direct-CI state has ranks `(12,66,220,66,12)` and 367,776 TT scalars. These are
representation/storage comparisons, not end-to-end speed or superiority
claims; direct CI remains cheaper and exact in this 924-dimensional truth
space.

## Backend boundary

The Blackwell complex128 AD probe passes, but the current Python-loop K4 GPU
workload exceeded the registered 600 s limit and was stopped. CPU is therefore
the production backend under ADR 0021. The largest completed point remains
below 108 s and 1 GB sampled RSS. GPU acceleration is unclaimed until a matched
vectorized implementation passes Phase 33.

## Reproduction

Run:

```powershell
.\.venv\Scripts\python.exe scripts/audit_phase32_n6_resources.py
.\.venv\Scripts\python.exe scripts/verify_phase32_n6_resource_audit.py
.\.venv\Scripts\python.exe scripts/benchmark_phase32_n6_convergence.py
.\.venv\Scripts\python.exe scripts/verify_phase32_n6_convergence.py
```

The verifier rebuilds Q128 Hamiltonians from raw exterior coefficients in the
committed artifact and does not use optimizer checkpoints.
