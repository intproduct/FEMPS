# Phase 15 report: ordered-distance Gate C

## Outcome

Gate C passes for the controlled finite-grid problem.  The ordered-sector
Hamiltonian now has an exact gap-coordinate MPS/MPO representation, a hard
finite-box charge sector, native latticeTN energy and AD, an explicit
polynomial contraction bound, and a random-initialization variational
reproduction of the `N=4` truth energy.  Neither production energy evaluation
nor training materializes a `d**(N+1)` state.

This decision is deliberately limited.  It does not establish continuum-basis
convergence, large-`N` accuracy, or novelty of distance-coordinate MPS.  It
authorizes the next continuous functional-basis bridge.

## Controlled problem

The benchmark uses four spinless fermions on eight centered coordinate-grid
points with spacing `0.7`, harmonic confinement, and

```text
V(x_i-x_j) = 1 / sqrt((x_i-x_j)^2 + 1).
```

The ordered sector has dimension `binom(8,4)=70`.  Its five nonnegative gap
variables sum to `G=4`; the complete local gap dimension is therefore five.

## Exact operator and representation checks

| Quantity | Result |
|---|---:|
| Exact finite-grid ground energy | `10.550426086401501` |
| Native exact-MPS/MPO energy error | `5.33e-15` |
| Exact gap-MPS internal ranks | `(5,10,10,5)` |
| Raw Hamiltonian MPO maximum bond | `33` |
| Native AD gradients at truth state | finite on all five cores |
| Production path gathers `d**(N+1)` | no |

Entrywise tests independently assemble the ordered-coordinate Hamiltonian and
verify the gap kinetic, harmonic, projector, and soft-Coulomb MPOs.  The
projector accepts exactly `sum g_i=4`.  CPU and RTX PRO 4000 Blackwell native
energies differ by `5.33e-15`, as do their largest absolute gradient entries.
The small problem is latency dominated: CPU takes `0.0133 s`, GPU takes
`0.0565 s`, and GPU peak allocation is `18.6 MB`; no speedup claim is made.

## Independent control axes

### Local gap cutoff

| `Q` | Sector dimension | Energy error vs complete `Q=4` | Native contraction error |
|---:|---:|---:|---:|
| 1 | 5 | `2.491` | `5.33e-15` |
| 2 | 45 | `4.968e-2` | `3.55e-15` |
| 3 | 65 | `1.078e-4` | `3.55e-15` |
| 4 | 70 | `0` | `5.33e-15` |

This separates local distance-basis truncation from MPS and MPO error.

### MPS bond

TT-SVD of the independent truth state gives the following diagnostic, with
charge weight exactly one in every row.

| Maximum bond | Retained ranks | Energy error |
|---:|---|---:|
| 1 | `(1,1,1,1)` | `3.914` |
| 2 | `(2,2,2,2)` | `9.367e-1` |
| 4 | `(4,4,4,4)` | `5.613e-3` |
| 8 | `(5,8,8,5)` | `1.535e-6` |
| 16 | `(5,10,10,5)` | `5.33e-15` |

This is a representation-capacity audit, not the variational training result.

### MPO compression

The raw bond-33 MPO is exact and is used for the blind optimization.  On an
independent smaller `Q=2` sector, maximum compressed bonds
`4,8,16,24,32` have relative Frobenius operator errors
`1.471e-1, 1.180e-1, 2.491e-2, 4.847e-5, 6.392e-15`, respectively.  Because
operator compression can shift the energy nonvariationally, a local discarded
singular-value norm is not treated as an error certificate.

### Grid and box

At fixed spacing `0.7`, enlarging `L=6,8,10,12` changes the energy through
`11.2439330, 10.5504261, 10.4463937, 10.4404120`.  At the fixed nominal box
`[-3.15,3.15]`, the pairs `(L,a)=(8,0.9),(10,0.7),(13,0.525),(15,0.45)` give
`9.8527364, 10.4463937, 10.7279274, 10.8230574`.  The latter sequence is not
yet converged and is reported to prevent finite-grid accuracy from being
mistaken for a continuum result.

## Blind variational optimization

The optimizer uses a hard-charge MPS whose virtual labels are cumulative gap
charges.  A per-charge correlation multiplicity of four gives bond dimensions
`(1,5,10,10,5,1)`.  Adam performs 2500 native Rayleigh-quotient steps on the
Blackwell GPU.  The exact matrix and eigenvector are constructed only after all
three runs finish.

The declared Gate C tolerance is absolute energy error `5e-5` for every seed.

| Seed | Initial energy | Optimized error | Ground-state fidelity | Charge weight | Time |
|---:|---:|---:|---:|---:|---:|
| 701 | `15.2205343` | `8.330e-6` | `0.999998590` | `1.0` | `26.45 s` |
| 1701 | `16.9016701` | `2.066e-5` | `0.999995682` | `1.0` | `25.17 s` |
| 2701 | `14.8636568` | `1.299e-6` | `0.999999811` | `1.0` | `25.00 s` |

Every forbidden parameter remains exactly zero.  Peak GPU allocation is at
most `19.7 MB`.  All three seeds pass the declared tolerance, so the result is
not a truth-state initialization or a selected successful restart.

## Complexity and prior-art boundary

With `S=N+1`, local dimension `d=Q+1`, total empty-site charge `G=L-N`, MPS
bond `chi`, and raw MPO bond `W`, the current direct-sum construction has

```text
W = O(N^2 G),
MPS storage = O(S d chi^2),
MPO storage = O(S d^2 W^2),
expectation <= O[S(W d chi^3 + W^2 d^2 chi^2)].
```

The hard-charge construction has `chi <= mu(G+1)` for requested per-charge
multiplicity `mu`.  These are representation-level polynomial bounds, not a
claim that the accuracy requirements for `Q` and `mu` are polynomial in all
physical regimes.

Li--Waintal is the direct prior method for ordered first-quantized
distance-variable MPS.  This phase therefore claims neither the ordered domain
nor the distance MPS as new.  The remaining project contribution must be
narrow: integrate this route with the 2201 orthonormal continuous-functional
operator/AD framework, provide controlled continuum benchmarks, and combine
that evidence with the matrix-wedge and ordinary-particle-TT obstruction
results.

## Gate C decision and next gate

Gate C is **PASS (finite-grid scope)** because:

1. the map, boundary, kinetic, trap, interaction, and constraint are exact on
   the finite grid;
2. the exact raw MPO and native AD contraction are polynomial in the explicit
   representation controls;
3. independent cutoff, MPS-bond, MPO-operator, box, and grid errors are
   separated;
4. three random hard-charge optimizations reproduce the `N=4` truth within the
   declared tolerance; and
5. the production path never gathers the exponential local tensor.

The next gate must replace gap-grid values by an orthonormal half-line
functional basis (plus a center-of-mass basis), verify continuous derivative
and interval-potential operators, and demonstrate basis/box/bond convergence.
Until that passes, the result remains a finite-grid ordered solver.

## Reproduction

```powershell
.\.venv\Scripts\python scripts\benchmark_ordered_distance_gate.py
.\.venv\Scripts\python scripts\gpu_smoke_ordered_distance.py
.\.venv\Scripts\python scripts\train_ordered_distance_n4.py
```

Machine-readable records are
`results/phase15_ordered_distance_gate.json`,
`results/phase15_ordered_distance_gpu_parity.json`, and
`results/phase15_ordered_distance_training.json`.
