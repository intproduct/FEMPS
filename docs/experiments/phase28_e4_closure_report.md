# Phase 28 E4 interacting FEMPS closure report

## Scope and evidence boundary

This report records **numerical evidence** for the restricted nonbranching
diagonal-path FEMPS,

\[
  \Psi=\sum_{a=1}^{K}c_a\,u_{a1}\wedge\cdots\wedge u_{aN}.
\]

It is a first-quantized continuous functional-basis matrix-wedge state and is
not a second-quantized MPS. It is also a nonorthogonal multideterminant/
selected-CI subclass, not an efficient contraction of generic FEMPS cores.

The benchmark and independent verifier are

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_phase28_e4_closure.py
.\.venv\Scripts\python.exe scripts\verify_phase28_e4_closure.py
.\.venv\Scripts\python.exe scripts\benchmark_phase28_diagonal_transition_scaling.py
```

Raw records are `results/phase28_e4_closure.json` and
`results/phase28_diagonal_transition_scaling.json`. No CI eigenvector is used
to initialize a FEMPS optimization. Full exterior and particle-tensor objects
are restricted to bounded post-run truth/comparator audits.

## Pre-registered E4 acceptance

Each stability run must have finite-basis energy error at most `1e-3`, energy
variance at most `1e-2`, norm error at most `1e-10`, and structural and
materialized antisymmetry residuals at most `1e-12`. It must use no enumerated
virtual paths. Fixed-`D` energy must be nonincreasing in `K=1,2,4`, fixed-`K=4`
continuum error must be nonincreasing in `D=5,6,7`, and every run must contain
sampled CPU resident-memory data.

| Stability group | Initialization | Passes | Maximum same-basis error | Maximum variance |
|---|---|---:|---:|---:|
| `N=4,D=6,K=4` | three blind seeded starts | 3/3 | `1.835e-12` | `2.272e-11` |
| `N=4,D=7,K=4` | three nested `D=6 -> D=7` continuations | 3/3 | `8.057e-5` | `1.011e-3` |

The continuation zero-pads an optimized `D=6` orbital set into the nested
`D=7` basis, exactly preserving its initial wavefunction. It does not use the
finite-basis ground-state vector. The independent verifier recomputes every
criterion from raw point data and returns `verified: true` and `E4_pass: true`.

## Independent `K` and `D` convergence

At fixed `N=4,D=6`:

| `K=chi` | Energy | Error vs finite-basis CI | Variance |
|---:|---:|---:|---:|
| 1 | 12.265072817277 | `5.249e-3` | `3.805e-2` |
| 2 | 12.259823862197 | `2.201e-11` | `2.995e-10` |
| 4 | 12.259823862177 | `1.382e-12` | `1.812e-11` |

At fixed `N=4,K=4`, using seed 17 on the reported axis:

| `D` | Energy | Absolute error vs continuum |
|---:|---:|---:|
| 5 | 12.590372160906 | `4.714e-1` |
| 6 | 12.259823862177 | `1.409e-1` |
| 7 | 12.173063670708 | `5.411e-2` |

The exact continuum energy is `12.118950038622252`. Both one-axis checks are
monotone. All reported FEMPS points have norm error below `5e-16`, zero
materialized and structural antisymmetry residuals, and zero enumerated
virtual paths.

## Comparators and the measured tradeoff

| Method at `D=6` | Energy | Same-basis error | Variance | Ordinary particle-TT ranks |
|---|---:|---:|---:|---|
| Single Slater / FEMPS `K=1` | 12.265072817277 | `5.249e-3` | `3.805e-2` | `(4,6,4)` |
| Single fixed-number AGP | 12.264881699154 | `5.058e-3` | `3.761e-2` | `(6,15,6)` |
| Diagonal-path FEMPS `K=2` | 12.259823862197 | `2.201e-11` | `2.995e-10` | `(6,15,6)` |
| Exact finite-basis CI | 12.259823862175 | 0 | `8.048e-30` | `(6,15,6)` |

At `D=7`, exact CI has energy `12.172983099708` and ordinary particle-TT
ranks `(7,21,7)`. The tested FEMPS `K=4` continuations lie `1.109e-5` to
`8.057e-5` above that truth.

This small model does not show a wall-time or raw-parameter advantage over
exact CI: the `D=6` exterior space has only 15 coefficients and direct
diagonalization takes about `0.067 s`. A `K=2` diagonal-path state stores 48
orbital scalars plus two amplitudes before gauge reduction. The useful result
is instead a measured structural tradeoff: exchange remains exact by
construction, `K` controls a systematic nonorthogonal correlation expansion,
and production observables use `K^2` transitions instead of a full particle
tensor. Whether this becomes advantageous when `binom(D,N)` is large remains
unproved and requires a larger nonquadratic benchmark.

## Time, memory, and transition implementation

The solver now samples whole-process CPU RSS every `5 ms`, including Torch
native allocations. Since all points run sequentially in one Python process,
absolute RSS includes the already loaded runtime and allocator cache; both
absolute peak and per-call increase are retained. The first `D=6,K=4` run
records `639,799,296` bytes peak RSS and `132,898,816` bytes above its baseline.
The three `D=7,K=4` runs take `23.35--23.65 s`, peak at
`665,247,744--689,217,536` bytes, and add `11.3--12.7 MB` over their respective
baselines. These sampled values are not claimed as isolated-process lower
bounds.

For nonsingular, well-conditioned determinant overlaps, the production path
uses determinant derivatives with linear solves. It automatically falls back
to column-replacement minors for singular or ill-conditioned pairs. Exact
tests cover both branches. Across ten `(N,D,K,L)` CPU points, the automatic
and forced-minor Hamiltonian matrices differ by at most `1.421e-14`. At
`N=4`, the automatic path gives roughly `2.1--3.6x` forward and `2.2--4.1x`
forward/backward speedups in the tested cases; at `N=6,D=8,K=4,L=1` the
speedups are `5.39x` and `7.41x`. At `N=2` the forward fast path is neutral,
which is reported rather than hidden.

The stored state size is `O(KDN)`. The measured kernel retains `K^2`
transition pairs; the well-conditioned factorized two-body path uses one
overlap factorization/solve per insertion and remains polynomial in
`K^2 L (D^2N+N^3)`. The singular-safe fallback performs the explicitly
recorded `K^2 L N(N-1)` determinant minors and is slower but exact.

## Decision

E4 passes the Phase 28 numerical gate. Together with E1--E3, this satisfies the
algorithm-recovery success criteria for the restricted diagonal-path subclass:
stable interacting optimization, independent `D`/`K` convergence,
reproduction and verification scripts, explicit variance/norm/symmetry/time/
memory fields, and no forbidden path or coefficient-tensor enumeration in
production.

The decision is deliberately narrow. It does not reopen a generic exact
contraction claim and does not establish novelty, asymptotic scalability, or
superiority to CI/AGP/DMRG. The next physics test should use a nonquadratic
soft-Coulomb interaction at a basis size where bounded exact CI is still
available for audit and then cross the direct-CI resource boundary only after
the controlled region is stable.
