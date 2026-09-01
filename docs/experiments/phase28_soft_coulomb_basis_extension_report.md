# Phase 28 soft-Coulomb basis-extension report

## Scope and scientific boundary

This report records **numerical evidence** for one checkpointed
`D=8 -> 10 -> 12` lineage of the restricted nonbranching diagonal-path FEMPS
at `N=4,K=4`. The Hamiltonian, continuous harmonic-oscillator functional
basis, and `Q=128` soft-Coulomb quadrature are the same as in the registered
transferability benchmark.

The calculation is first quantized and antisymmetric by exterior construction.
It is neither a generic FEMPS contraction nor a second-quantized MPS. The
production contraction does not enumerate virtual paths or materialize the
full particle coefficient tensor. Dense exterior CI and particle-tensor
materialization are bounded validation tools only.

Reproduce and independently verify the recorded artifact with

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_phase28_soft_coulomb_transferability.py
.\.venv\Scripts\python.exe scripts\benchmark_phase28_soft_coulomb_basis_extension.py
.\.venv\Scripts\python.exe scripts\verify_phase28_soft_coulomb_basis_extension.py
```

Raw results are in
`results/phase28_soft_coulomb_basis_extension.json`; checkpoints are ignored
reproduction artifacts. No CI eigenvector is used to initialize FEMPS.

## Registered acceptance

The extension retained the earlier thresholds without relaxation:

- same-basis dense-CI energy error at most `2e-4`;
- energy variance at most `2e-3`;
- norm error at most `1e-10`;
- structural and materialized antisymmetry residuals at most `1e-12`;
- dense operator-factorization error at most `1e-11`;
- zero enumerated virtual paths and a measured process peak RSS;
- nonincreasing absolute error against the pre-existing `D=14` numerical
  reference along `D=8,10,12` at fixed `K=4`.

Both extension points and the complete basis-axis condition pass. The
independent verifier recomputes the acceptance decision from the raw fields.

## State and truth audit

The accepted seed-17 `D=8,K=4` checkpoint was zero-padded into `D=10`; the
optimized `D=10` state was then zero-padded into `D=12`. Each point used 80
Adam steps and 40 L-BFGS refinement steps.

| `D` | Exterior dimension | FEMPS energy | Dense-CI energy | Same-basis error | Variance |
|---:|---:|---:|---:|---:|---:|
| 10 | 210 | 11.023237957891551 | 11.023133765392014 | `1.041925e-4` | `9.668480e-4` |
| 12 | 495 | 11.023177795943152 | 11.023094656411180 | `8.313953e-5` | `9.014082e-4` |

The maximum norm error is `3.331e-16`. Structural and materialized
antisymmetry residuals are exactly zero at both points. The direct reference
uses the unfactorized quadrature tensor and an independently diagonalized
exterior Hamiltonian.

## Basis convergence

The `D=14` energy `11.023082853674637` is a finite-basis **numerical
reference**, not a continuum theorem or bound.

| `D` | `K` | FEMPS energy | Absolute error versus `D=14` reference |
|---:|---:|---:|---:|
| 8 | 4 | 11.023424901566514 | `3.420479e-4` |
| 10 | 4 | 11.023237957891551 | `1.551042e-4` |
| 12 | 4 | 11.023177795943152 | `9.494227e-5` |

The registered absolute error is nonincreasing. The separate finite-basis CI
error at `D=12` is `1.180274e-5`; the remaining FEMPS optimization/correlation
error is therefore visible rather than being folded into the basis error.

## Operator and cost audit

The physical operator-SVD backend retains ranks 19 and 23 at `D=10` and
`D=12`, with dense reconstruction errors `1.340e-15` and `3.847e-15`.

| `D` | Total FEMPS call time | Sampled peak RSS | FEMPS particle-TT ranks | Dense-CI particle-TT ranks |
|---:|---:|---:|---:|---:|
| 10 | 60.08 s | 706,297,856 bytes | `(10,24,10)` | `(10,45,10)` |
| 12 | 92.77 s | 823,091,200 bytes | `(12,24,12)` | `(12,66,12)` |

Dense CI took 4.56 s and 16.21 s for these bounded truth calculations, so the
result does not establish a time or memory advantage. The measured structural
tradeoff is that exchange antisymmetry remains exact while the FEMPS
correlation control `K=4` keeps the central ordinary particle-TT rank at 24;
the corresponding dense-CI ranks grow to 45 and 66. Production nevertheless
pays the explicit `K^2 L` transition cost, and fixed `K` leaves a measurable
correlation error.

## Decision

The registered basis-extension gate passes. Together with the prior three-seed
and `K=1,2,4` audits, this closes the minimum interacting algorithm-and-physics
recovery criterion for the restricted diagonal-path route: stable optimization,
independent reproduction/verification, controlled finite-basis errors and
variance, exact reported antisymmetry, systematic `D` and `K` behavior, and no
virtual-path enumeration.

This is not a generic scalability or superiority result. The next work is
method consolidation: one clean-process reproduction of the complete lineage
and a high-basis correlation check that separates fixed-`K` error from basis
error. Larger `N` is deferred until that bounded decision is recorded.
