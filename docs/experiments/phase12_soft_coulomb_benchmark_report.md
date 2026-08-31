# Phase 12 controlled soft-Coulomb benchmark matrix

## Scope

This phase benchmarks the fixed-number finite-AGP subclass admitted by the
CONDITIONAL Gate A decision. It does not promote generic matrix-wedge FEMPS to
a polynomially contractible method.

Every normalized point records `N,D,K,Q`, seed, energy, direct truth,
correlation/optimizer error, operator-representation error, basis error, total
error, fidelity, raw and balanced overlap conditions, retained/discarded rank,
explicit exterior agreement, wall time, peak memory, and pruning/restart
events. The signed error identity is

`correlation + operator + basis = total`.

The direct dense exterior truth boundary is fixed at
`binom(D,N) <= 1200`; larger points must not silently use the dense oracle.

## Independent operator and basis truth

At the largest basis tested for each particle number, the direct dense
Slater--Condon energy change from `Q=128` to `Q=160` is:

| N | D | Exterior dimension | Q128-Q160 |
|---:|---:|---:|---:|
| 4 | 12 | 495 | -5.51e-14 |
| 6 | 12 | 924 | -2.91e-13 |
| 8 | 12 | 495 | -9.45e-13 |

The `Q=128` direct basis references are:

| N | D | Ground energy | Difference from largest D |
|---:|---:|---:|---:|
| 4 | 10 | 11.023133765392 | 5.091e-5 |
| 4 | 12 | 11.023094656411 | 1.180e-5 |
| 4 | 14 | 11.023082853675 | 0 |
| 6 | 8 | 25.051232892768 | 1.866e-3 |
| 6 | 10 | 25.049639839832 | 2.734e-4 |
| 6 | 12 | 25.049366416097 | 0 |
| 8 | 10 | 44.451700103591 | 5.691e-3 |
| 8 | 12 | 44.446009528434 | 0 |

The largest D values are numerical references, not continuum bounds.

## N=4: independent D and K control

At `D=10,K=5`, seeds 301--303 have finite-basis errors
`6.00e-6`, `8.82e-6`, and `8.03e-6`; total errors relative to the D14
numerical reference are `5.69e-5`--`5.97e-5`.

Embedding the same three chains in `D=12,K=5` gives finite-basis errors
`1.46e-5`, `1.84e-5`, and `1.53e-5`, but the smaller basis error lowers total
errors to `2.64e-5`--`3.02e-5`. A `D=12,K=6` probe lowers the seed-301
finite-basis error from `1.462e-5` to `7.923e-6`, and its total error to
`1.973e-5`. Thus increasing D and K are distinct variational controls.

## N=6: K must grow after enlarging D

The legacy `D=8,K=2` chain has finite-basis error `1.30e-6`, but its total
error relative to D12 is dominated by the `1.866e-3` basis error. At D10, K2
has finite-basis error `5.47e-4`; adding K3 lowers it to
`1.031e-4`, `1.088e-4`, and `1.109e-4` for three independently seeded third
terms sharing the same converged K2 prefix. Total errors become
`3.77e-4`--`3.84e-4`.

The apparent old D8/K2 overlap condition `26.1` is another term-norm gauge
artifact: its balanced condition is `1.03`. Final D10/K3 balanced conditions
are `2.46`--`2.62`.

## N=8: reproducible K hierarchy and D/K interaction

At D10, one blind chain gives:

| K | Finite-basis error | Balanced condition |
|---:|---:|---:|
| 1 | 2.084e-3 | 1.000 |
| 2 | 3.129e-5 | 1.047 |
| 3 | 7.333e-7 | 1.073 |

Two independently seeded K3 extensions from the common K2 prefix reach
`3.594e-7` and `1.415e-6`, with balanced conditions `1.156` and `1.089`.
The K2-to-K3 improvement is therefore reproducible at the measured budgets.

At D12, the embedded K3 chain has finite-basis error `6.158e-4`: the absolute
energy still improves strongly because the D10 basis error was `5.691e-3`, but
fixed K3 cannot exploit all new orbital freedom. This is negative evidence
against treating a fixed K as basis-independent.

## Matched representation comparison

A separate N4 finite-difference grid uses exactly the same one-body and
diagonal soft-Coulomb pair operators for all routes. At D=8:

- ordered-sector and direct Slater--Condon exterior Hamiltonians agree within
  `1.07e-14`;
- finite-AGP K=1,2,3 errors are `3.037e-2`, `8.309e-5`, and `1.204e-5`;
- polynomial and explicit exterior finite-AGP energies agree within
  `5.87e-14`;
- the explicit particle tensor has ordinary TT ranks `[8,28,8]` at every K.

For D=8,10,12 grid truths, ordinary particle-TT ranks grow as
`[8,28,8]`, `[10,45,10]`, and `[12,63,12]`. The ordered-sector vector avoids
the N! labeled copies but still has `binom(D,N)` elements and remains an
exponential truth oracle here, not a production tensor network.

## Stability and cost

No production point discarded, pruned, or restarted an AGP direction. All
reported balanced conditions remain below `3.16`. The worst polynomial versus
explicit exterior energy mismatch in the normalized matrix is about `8.3e-12`
at N8; the error is small relative to the reported variational errors but must
remain a regression target.

Representative final-stage Blackwell measurements range from about `39 s` and
`43 MB` for N6/D10/K2 to `241 s` and `226 MB` for N4/D12/K6. These are measured
points, not an asymptotic scaling claim.

## Novelty correction and decision

The literature audit found that Uemura--Kasamatsu--Sugino (2015) already use
linear combinations of independently optimized AGPs/HFB states with
deterministic matrix elements and quadratic term-count cost. Dutta et al.
(2021) develop linearly independent nonorthogonal AGP sets, and
Kawasaki--Gao--Scuseria (2026) rewrite inter-geminal AGP-CI as compact LC-AGP.

Consequently, finite AGP is a validated polynomial fallback and benchmark
control, not by itself a novel FEMPS ansatz. Phase 12 passes its numerical exit
criterion, but the evidence does **not** justify a paper centered on LC-AGP.
The next phase must return to a structure beyond standard LC-AGP: a nontrivial
matrix-wedge family with polynomial contraction, or a proved
statistics-carrier/correlation-multiplicity factorization.

The normalized table is
`results/phase12_benchmark_manifest.json`; direct truth, raw training, ordered
sector, and matched-grid artifacts are stored alongside it in `results/`.
