# Phase 9 soft-Coulomb operator report

## Convention

The first E6 model is a spin-polarized two-fermion system on `R`, in oscillator
units, with trap frequency one and repulsive pair potential

\[
v(x,y)=\frac{g}{\sqrt{(x-y)^2+a^2}},\qquad g=a=1.
\]

This is a controlled one-dimensional electronic-like benchmark, not a
realistic electronic-structure model.

## Quadrature and factorization

At probe basis `D=8`, direct Gauss--Hermite four-index tensors converge against
`Q=128` as follows:

| `Q` | Relative direct-quadrature error |
|---:|---:|
| 24 | 8.076e-5 |
| 32 | 4.192e-6 |
| 48 | 9.720e-8 |
| 64 | 4.219e-9 |
| 96 | 2.246e-11 |
| 128 | 0 |

The weighted soft-Coulomb kernel is diagonalized into one-body operator
factors. At `Q=128,D=8`, the rank/error tradeoff is:

| Relative eigenvalue threshold | Retained rank | Dense tensor error |
|---:|---:|---:|
| 1e-8 | 32 | 4.784e-5 |
| 1e-10 | 38 | 4.383e-7 |
| 1e-12 | 44 | 6.606e-10 |
| 1e-14 | 50 | 6.543e-13 |
| 0 | 128 | 5.350e-13 |

The `1e-14` threshold is therefore used for the first production benchmark.
The residual at zero threshold reflects numerical eigendecomposition and dense
reconstruction roundoff, not discarded modes. Hermiticity and particle-exchange
residuals remain at double-precision roundoff.

## Independent N=2 energy checks

Exact diagonalization in the antisymmetric HO sector at `Q=128` gives:

| `D` | Finite-basis energy | Difference from `D=16` |
|---:|---:|---:|
| 4 | 2.554288734101 | 4.569e-4 |
| 6 | 2.553885908134 | 5.409e-5 |
| 8 | 2.553842079144 | 1.026e-5 |
| 10 | 2.553834336261 | 2.517e-6 |
| 12 | 2.553832510344 | 6.915e-7 |
| 14 | 2.553831989221 | 1.704e-7 |
| 16 | 2.553831818868 | 0 |

An independent ordered half-line calculation separates center-of-mass and odd
relative motion. Dirichlet finite differences on `r in (0,8)` at spacings
`0.1, 0.0667, 0.0444, 0.0333` converge upward; second-order extrapolation from
the last two gives `2.553831733979`. The difference from HO `D=16` is
`8.49e-8`. At fixed spacing `1/30`, changing the half-width from `6` to `8` to
`10` changes the energy by at most `5.1e-13`, so the displayed grid discrepancy
is discretization rather than box leakage.

## Polynomial/exterior validation

For a random complex skew pair matrix at `D=5,Q=48`, the factorized polynomial
AGP energy and its AD gradient agree with an explicitly materialized exterior
Hamiltonian within the test tolerances (`2e-11` for energy and `3e-10` for the
raw skew gradient). The factorized interaction itself agrees with the direct
four-index quadrature path at `2e-13` elementwise tolerance.

These tests validate the operator and contraction interface. Blind/restarted
variational optimization remains the next checkpoint; the `D=16` and grid
extrapolations are numerical references, not rigorous continuum bounds.

Raw evidence is in `results/soft_coulomb_operator_sweep.json`.

## Blind and restarted N=2 optimization

A blind complex skew pair matrix at `D=12,Q=128` was optimized for 200 steps,
saved, and resumed to 800 steps on RTX PRO 4000 Blackwell. The restart retained
the optimizer and cosine-scheduler state. Results are:

| Quantity | Value |
|---|---:|
| Initial energy | 12.185311535712 |
| Final polynomial energy | 2.553832510344195 |
| Finite-basis truth | 2.553832510344217 |
| Signed finite-basis difference | -2.13e-14 |
| Polynomial/exterior difference | 2.09e-14 |
| Finite-basis fidelity | 0.9999999999999998 |
| Resumed 600-step time | 490.89 s |
| Peak CUDA allocation | 19,161,600 bytes |

The tiny negative signed difference is roundoff, not a variational violation.
The result passes the N=2 accuracy and restart check. Runtime is the important
negative result: the current general contraction handles 50 soft-Coulomb
operator factors sequentially and is too slow to scale directly to N=4.
Batching/fusing the factor axis is therefore required before the N=4 run.

Raw evidence is in `results/soft_coulomb_n2.json`; the checkpoint is deliberately
ignored by Git.

## Batched contraction and N=4 benchmark

The factor loop was replaced by a batched second-order Newton recurrence that
propagates the two first derivatives and their mixed derivative together. It
does not invoke nested autograd separately for each operator factor. Existing
explicit energy and gradient tests remain unchanged and pass.

For `N=2,D=12,Q=128`, measured per-step time falls from `0.818 s` in the
sequential 600-step segment to `0.0411 s` in a 100-step batched timing run, a
19.9-fold improvement, with no material increase in peak memory.

The first `N=4,D=8,Q=96,K=1` run was interrupted at step 150 and resumed to
step 600:

| Quantity | Value |
|---|---:|
| Initial energy | 19.471133728926 |
| Final energy | 11.024724780728 |
| Finite-basis truth | 11.023278984745 |
| Finite-basis error | 1.446e-3 |
| Polynomial/exterior difference | 1.60e-14 |
| Finite-basis fidelity | 0.9996928175 |
| Resumed 450-step time | 19.62 s |
| Peak CUDA allocation | 18,071,040 bytes |

This passes the blind/restart and independent-contraction checks, but the
single-AGP representation error is still material. Finite-AGP greedy growth is
the next step before attempting `N=6`.

Raw evidence is in `results/soft_coulomb_n4.json`.

Greedy `K=2` growth lowers the N=4 error further:

| Quantity | K=2 joint value |
|---|---:|
| Finite-basis error | 7.445e-5 |
| Polynomial/exterior difference | 1.39e-13 |
| Finite-basis fidelity | 0.9999904819 |
| Retained overlap condition number | 1.393 |
| Generalized residual | 9.57e-16 |
| Joint 300-step time | 33.10 s |
| Peak CUDA allocation | 27,449,856 bytes |

At `N=6,D=8,Q=96,K=1`, 600 blind steps take `33.68 s` and reach
`1.725e-3` finite-basis error with fidelity `0.9996265`. Polynomial and
independent 28-dimensional exterior energies differ by `1.19e-12`; no ordinary
`D^6` tensor is needed by the optimization.

Raw evidence is in `results/soft_coulomb_n4_greedy_k2.json` and
`results/soft_coulomb_n6.json`.

## Phase decision

E6 passes its stated Phase 9 exit criterion. The operator has independently
controlled quadrature/factorization error; N=2 and N=4 have finite-sector truth,
restart evidence, and polynomial/exterior agreement; the safe N=6 attempt also
passes contraction checks. The next phase should map the `D,K,N` correlation
hierarchy and improve finite-AGP growth before any higher-dimensional or
realistic-electronic claim.
