# Phase 6 report: conditioned finite-AGP optimization

## Linear amplitude elimination

For fixed pair matrices, the solver now forms the polynomial matrices

\[
 S_{ab}=\langle\Phi_a|\Phi_b\rangle,\qquad
 H_{ab}=\langle\Phi_a|H|\Phi_b\rangle
\]

and solves `Hc=ESc` in the retained positive overlap subspace. Tests compare
every matrix element with independent exterior coefficients, verify
Hermiticity, and confirm that duplicate AGPs reduce automatically from nominal
`K=2` to effective rank one.

Each nonlinear step therefore optimizes only pair matrices. Pair Frobenius
scales and anchor phases are fixed in the forward map; near-dependent span
directions are thresholded; output terms receive a deterministic permutation.
Every record includes the overlap spectrum, effective rank, condition number,
and generalized residual. Checkpoint/resume and best-state restoration are
covered by deterministic regression.

## Why simultaneous blind K=2 still fails

At `D=8,kappa=0.35`, 600 steps starting from two random AGPs give

| Quantity | Value |
|---|---:|
| Initial energy | 25.728659509649 |
| Final error vs finite truth | 2.905e-3 |
| Overlap condition number | 1.909 |
| Generalized residual | 1.83e-15 |
| Ground-state fidelity | 0.9994133 |

The trajectory plateaus smoothly after about 120 steps. The small condition
number and residual rule out a failed amplitude solve: the remaining obstacle
is nonlinear specialization of the two geminals.

## Greedy growth without exact-state information

The accepted strategy is:

1. optimize a blind `K=1` state;
2. freeze it and add one random AGP;
3. optimize the new AGP using the exact `2x2` generalized amplitude solve;
4. unfreeze both pair matrices and jointly relax them.

No exact eigenvector or oracle pair matrix enters this path. Three independent
new-AGP seeds give:

| Seed | Error after frozen growth | Error after joint relaxation | Condition number | Residual |
|---:|---:|---:|---:|---:|
| 0 | 1.075e-4 | 3.110e-5 | 1.398 | 1.13e-15 |
| 1 | 1.034e-4 | 2.336e-5 | 1.325 | 1.34e-15 |
| 2 | 9.753e-5 | 2.000e-5 | 2.409 | 5.09e-16 |

Polynomial and independent exterior energies agree within `1.1e-13`. The
final errors lie within `2.1e-5` of the separately oracle-initialized `K=2`
energy and have a `1.11e-5` seed spread. This is the documented Phase 6 blind
acceptance tolerance; it is not yet chemical-accuracy terminology.

## Basis-size repeat

The oracle representation diagnostic was repeated at `D=10,kappa=0.35`:

| `K` | Error vs finite truth | Ordinary internal particle-TT ranks |
|---:|---:|---:|
| 1 | 2.719e-3 | `(10,45,10)` |
| 2 | 6.238e-6 | `(10,45,10)` |
| 4 | 3.778e-6 | `(10,45,10)` |
| 8 | 7.784e-7 | `(10,45,10)` |

The `D=10` finite-basis continuum error is `1.024e-3`, so already `K=2`
pushes representation error far below basis error. Polynomial/exterior
differences remain below `7.2e-15`.

## Decision

Phase 6 passes. Greedy variable projection replaces simultaneous amplitude/pair
Adam as the default finite-AGP growth method. The next phase may increase
particle number, while retaining the rule that a new AGP is added and tested
before all terms are jointly released.

Raw evidence is in `results/fermion_e4_variable_projection.json`,
`results/fermion_e4_greedy_k2.json`, and
`results/fermion_e4_agp_rank_sweep_d10.json`.
