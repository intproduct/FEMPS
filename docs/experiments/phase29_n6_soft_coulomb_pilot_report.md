# Phase 29 resource-capped N=6 soft-Coulomb pilot

## Scope and preregistration

ADR 0018 admits exactly one truth-controlled larger-particle pilot for the
restricted diagonal-path FEMPS:

\[
 H=\sum_{i=1}^{6}\left(-\frac12\partial_{x_i}^2+\frac12x_i^2\right)
 +\sum_{i<j}\frac{1}{\sqrt{(x_i-x_j)^2+1}},
 \qquad D=10,\ Q=128.
\]

The state is first quantized in continuous harmonic-oscillator functional
bases. `K=1` is a blind Slater optimization. The `K=4` initial span preserves
that determinant exactly and adds three seed-2914 blind Slaters; no CI
eigenvector is used in initialization.

The experiment is **single-seed numerical feasibility evidence**, not a
stability or scaling claim. Reproduce and verify with

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_phase29_n6_soft_coulomb_pilot.py
.\.venv\Scripts\python.exe scripts\verify_phase29_n6_soft_coulomb_pilot.py
```

Raw data are in `results/phase29_n6_soft_coulomb_pilot.json`.

The preregistered limits were 1.5 GiB sampled process RSS and 600 s per point,
operator/reference disagreement below `1e-11`, norm error below `1e-10`, both
antisymmetry residuals below `1e-12`, and zero virtual-path enumeration. The
`K=4` point additionally required same-basis CI error at most `5e-4`, no more
than half the `K=1` error, and variance at most `5e-3`.

## Independent truth and operator audit

The exterior dimension is only `binom(10,6)=210`, so the reference diagonalizes
the direct unfactorized quadrature Hamiltonian. Its energy is
`25.049639839832263`, variance `9.20e-29`, norm error `6.66e-16`, and
materialized antisymmetry residual zero. The one-million-coefficient particle
tensor has ordinary TT ranks `(10,45,120,45,10)`.

The production physical-operator SVD has rank 19 and dense reconstruction error
`1.340e-15`. Factorized and direct finite-basis reference energies agree within
the registered `1e-11` tolerance.

## FEMPS result

| `K` | Energy | Same-basis CI error | Variance | Total time | Peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.052242782725010 | `2.602943e-3` | `1.477540e-2` | 16.59 s | 706,211,840 bytes |
| 4 | 25.049778081825440 | `1.382420e-4` | `1.196813e-3` | 85.59 s | 798,416,896 bytes |

The `K=4` error is 5.31% of the `K=1` error, a 94.69% reduction, and is below
the absolute `5e-4` gate. The nested `K=4` initial energy is
`25.052214571919464`, already no worse than the source; optimization lowers it
further. All four determinant directions are retained with overlap condition
number 4.815.

The `K=4` norm error is `4.44e-16`; structural and million-coefficient
materialized antisymmetry residuals are zero. Production stores 240 orbital
scalars, evaluates 16 transition pairs and 9,120 factorized two-body minors,
and enumerates zero virtual paths. Its ordinary particle-TT ranks are
`(10,45,80,45,10)`, compared with `(10,45,120,45,10)` for dense CI.

## Interpretation and decision

Every ADR 0018 acceptance and resource condition passes under the independent
verifier. This extends the restricted FEMPS feasibility evidence from N4 to one
N6 interacting point while retaining direct CI, variance, materialization,
symmetry, timing, memory, and comparator audits.

The measured tradeoff remains explicit. FEMPS separates exact exterior
exchange structure from a `K`-controlled correlation span and reduces the
central particle-TT rank at this point, but direct CI is still faster in the
small truth space and production cost grows as `K^2 L`. One seed is not
optimization-stability evidence.

No N8 or asymptotic claim is admitted. The next gate is a bounded N6
reproducibility decision: either run preregistered multiseed stability at this
same `(N,D,K)` or stop the diagonal-path size extension if the required cost is
not justified. It must not combine a new particle number, basis order and K
growth in one uncontrolled step.
