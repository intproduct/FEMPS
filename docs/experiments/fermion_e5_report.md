# E5 report: six-fermion particle-number scaling

## E5a noninteracting representation benchmark

Six spin-polarized fermions occupy oscillator functions `n=0,...,5`, giving

\[
 E_0=\frac{N^2}{2}=18.
\]

At `D=8`, the direct Slater tensor, correlation-bond-one matrix-wedge FEMPS,
three-pair-channel Pfaffian state, ordered-channel exterior coefficients, and
independent exterior Hamiltonian all agree exactly in float64 arithmetic.

The ordinary particle-TT ranks, including boundaries, are

\[
 (1,6,15,20,15,6,1),
\]

while all five direct FEMPS correlation bonds are one. The nonzero particle
Schmidt values are flat with multiplicities `6,15,20,15,6`; at the central cut
the best ordinary rank-19 approximation still has relative error
`1/sqrt(20)`.

A blind polynomial Blackwell run gives:

| Quantity | Value |
|---|---:|
| Initial energy | 23.916452885705 |
| Final energy | 18.000000000000 |
| Finite-basis error | 1.78e-13 in magnitude |
| Ground-state fidelity | 1 within roundoff |
| Steps | 800 |
| GPU time | 17.32 s |
| Peak CUDA allocation | 18,262,016 bytes |

No `D^6` tensor is formed during optimization. It is materialized only after
training for the safe `D=8` rank-spectrum oracle.

## E5b interacting truth

For six particles with all-to-all harmonic interaction,

\[
 E_{\rm cont}=\frac12+\frac{35}{2}\sqrt{1+6\kappa}.
\]

Finite exterior-sector errors against this continuum value are:

| `D` | `kappa=0.05` | `kappa=0.1` | `kappa=0.35` |
|---:|---:|---:|---:|
| 6 | 1.719e-1 | 6.141e-1 | 5.063e0 |
| 8 | 4.949e-3 | 4.985e-2 | 1.408e0 |
| 10 | 7.867e-5 | 2.565e-3 | 3.554e-1 |
| 12 | 8.659e-7 | 9.194e-5 | 7.419e-2 |

The first production point uses `D=10,kappa=0.1`, where the exterior truth
dimension is `binom(10,6)=210` and the basis error is `2.565e-3`.

## Single and greedy two-AGP results

Three blind single-AGP seeds give finite-basis errors

\[
 2.473\times10^{-4},\quad
 2.458\times10^{-4},\quad
 2.510\times10^{-4},
\]

with a spread of only `5.2e-6`. For seed zero, polynomial and independent
exterior energies differ by `1.28e-13`; GPU training takes `86.6 s` and peaks
at `19.2 MB`. Its post-training ordinary internal TT ranks are
`(10,44,110,44,10)`.

Starting from the blind seed-zero K=1 state, freezing it and greedily optimizing
one random new AGP lowers the finite-basis error to `4.765e-6`. The subsequent
joint Adam pass drifts upward, so best-state restoration correctly retains the
greedy result. Diagnostics are:

| Quantity | K=2 greedy value |
|---|---:|
| Finite-basis error | 4.765e-6 |
| Continuum error | 2.570e-3 |
| Ground-state fidelity | 0.9999992430 |
| Overlap condition number | 2.731 |
| Generalized residual | 2.48e-16 |
| Polynomial/exterior difference | 1.03e-13 |
| Ordinary internal particle-TT ranks | `(10,45,120,45,10)` |
| Growth time | 82.10 s |
| Peak CUDA allocation | 22,316,544 bytes |

Thus K=2 representation error is over 500 times smaller than the oscillator
basis error, while the explicit ordinary particle tensor reaches the full
antisymmetric structural ranks at every cut. This is measured only at the
reported `N,D,K`; it is not extrapolated to asymptotic scaling.

## Decision

Phase 7 passes. The exterior/Pfaffian control remains compact and accurately
contractible at six particles, and greedy growth supplies a practical
correlation improvement. Before soft-Coulomb E6, the next phase will add an
ordered-sector comparison and one non-materialized larger-`N` point so that the
method comparison is not based solely on ordinary particle TT.

Raw evidence is in `results/fermion_e5a.json`,
`results/fermion_e5b_truth_sweep.json`,
`results/fermion_e5b_single_agp.json`, and
`results/fermion_e5b_greedy_k2.json`.
