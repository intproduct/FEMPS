# Phase 29 matched N4-to-N6 contraction-cost audit

## Scope

This audit isolates particle-count effects in the accepted diagonal-path FEMPS
kernel. It uses optimized soft-Coulomb checkpoints at fixed `D=10,K=4`, the
same physical operator-SVD rank `L=19`, and the same CPU value/reverse-mode
operations. No optimizer loop, CI diagonalization, or materialization is inside
the timed region.

The result is **two-point numerical cost evidence**, not asymptotic scaling.
Reproduce and verify with

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_phase29_n4_n6_matched_cost.py
.\.venv\Scripts\python.exe scripts\verify_phase29_n4_n6_matched_cost.py
```

Raw data are in `results/phase29_n4_n6_matched_cost.json`.

## Exact structural counts

| Quantity | N4 | N6 | N6/N4 |
|---|---:|---:|---:|
| Stored orbital scalars `KDN` | 160 | 240 | 1.5 |
| Transition pairs `K^2` | 16 | 16 | 1.0 |
| One-body determinants `K^2 N` | 64 | 96 | 1.5 |
| Two-body determinants `K^2 L N(N-1)` | 3,648 | 9,120 | 2.5 |
| Enumerated virtual paths | 0 | 0 | -- |
| Production particle coefficients | 0 | 0 | -- |

Both checkpoints use the well-conditioned inverse path for all 16 transition
pairs. Auto and singular-safe minor Hamiltonians agree within `1.07e-14`.

## Matched timings

Each entry is the median of five warmed CPU evaluations.

| Kernel | N4 | N6 | N6/N4 |
|---|---:|---:|---:|
| Auto value | 0.09252 s | 0.11462 s | 1.239 |
| Auto value + reverse mode | 0.29904 s | 0.30623 s | 1.024 |
| Minor value | 0.39405 s | 0.83342 s | 2.115 |
| Minor value + reverse mode | 1.22434 s | 2.73328 s | 2.232 |

The minor ratios track the exact 2.5 two-body determinant-count growth within
small-kernel overhead. The vectorized inverse path grows less over these two
points because fixed launch/batching work is still significant. This must not
be extrapolated as constant cost in N.

Relative to minors, the inverse-fast path accelerates N4 value/gradient by
4.26x/4.09x and N6 by 7.27x/8.93x. The N6 minor reverse-mode measurement has a
101 MB sampled RSS increase versus about 4 MB for auto. Absolute sampled peaks
remain process-level measurements with allocator history, not isolated lower
bounds.

## Exchange carrier and correlation structure

At fixed FEMPS correlation multiplicity `K=4`:

| State | N4 particle-TT ranks | N6 particle-TT ranks |
|---|---|---|
| Diagonal-path FEMPS | `(10,24,10)` | `(10,45,80,45,10)` |
| Dense CI | `(10,45,10)` | `(10,45,120,45,10)` |

The two exterior truth spaces both have dimension 210 because
`binom(10,4)=binom(10,6)`. That accidental equality makes this a useful
controlled comparison: FEMPS stores only `KDN` orbital scalars while the
ordinary particle representation exposes the growing exchange-cut
multiplicity. The center rank still grows from 24 to 80, so exterior structure
does not remove the physical exchange carrier; it prevents that carrier from
being misidentified as the correlation control `K`.

## Decision

The independent verifier accepts value agreement, operator reconstruction,
memory records, and exact structural ratios. Together with the three-seed N6
physics gate, Phase 29 closes successfully at its bounded scope.

N expansion stops. Direct CI remains faster in the current 210-dimensional
truth region, while diagonal-path FEMPS provides a polynomial first-quantized
continuous solver with exact exchange structure and systematic K correlation.
The next phase is method consolidation and external-control packaging, not an
N8 run or a return to high-dimensional rank classification.
