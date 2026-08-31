# Phase 10 soft-Coulomb correlation hierarchy

## Independent operator and basis axes

For `N=4,D=8`, direct exterior truth changes by only `-4.79e-12` between
Gauss--Hermite orders `Q=96` and `Q=128`. Operator quadrature is therefore
negligible at the production settings.

At `Q=128`, the independent direct-four-index Slater--Condon truth is:

| `D` | Exterior dimension | Ground energy | Difference from `D=14` |
|---:|---:|---:|---:|
| 4 | 1 | 11.085944151108 | 6.286e-2 |
| 6 | 15 | 11.023837713203 | 7.549e-4 |
| 8 | 70 | 11.023278984750 | 1.961e-4 |
| 10 | 210 | 11.023133765392 | 5.091e-5 |
| 12 | 495 | 11.023094656411 | 1.180e-5 |
| 14 | 1001 | 11.023082853675 | 0 |

`D=14` is still only a numerical reference, not a continuum bound.

## Finite-AGP K hierarchy

At `D=8,Q=96`, one no-oracle greedy chain gives:

| `K` | Finite-basis error | Fidelity | Overlap condition |
|---:|---:|---:|---:|
| 1 | 1.446e-3 | 0.9996928 | 1 |
| 2 | 7.445e-5 | 0.9999905 | 1.393 |
| 3 | 1.956e-5 | 0.9999976 | 2.013 |
| 4 | 5.407e-6 | 0.9999993 | 2.081 |

Three fully independent blind `K=1 -> K=4` chains, using a shorter common
budget, finish at `9.190e-6`, `9.625e-6`, and `1.271e-5`. Their spread is
`3.52e-6`; retained overlap conditions are `5.94`, `9.04`, and `12.76`, with
no discarded directions and generalized residuals near `1e-15`.

At `D=10,Q=128`, one chain gives errors `1.805e-3`, `1.540e-4`, `5.595e-5`,
and `2.040e-5` for `K=1,2,3,4`. Absolute variational energy nevertheless drops
below every D=8 result. The K=4 overlap condition rises to `143.5`, but all four
directions remain retained, the generalized residual is `1.02e-15`, and the
polynomial/exterior difference is `8.70e-14`.

## Error decomposition

Using `D=14,Q=128` truth only as the largest computed reference:

| Point | Operator error | Basis error | K/optimizer error | Total vs D=14 |
|---|---:|---:|---:|---:|
| D=8, K=4 | 4.8e-12 | 1.961e-4 | 5.407e-6 | 2.015e-4 |
| D=10, K=4 | below Q128 resolution | 5.091e-5 | 2.040e-5 | 7.131e-5 |

The data demonstrate variational improvement in both D and K, while also
showing that basis and nonlinear-optimizer errors are now comparable at D=10.

## N=6 and contraction cost

At `N=6,D=8,Q=96`, greedy K=2 lowers the finite-basis error from `1.725e-3`
to `1.484e-6`, with fidelity `0.99999968`, overlap condition `26.1`, and
polynomial/exterior agreement `1.17e-12`.

A matched current-kernel `N=4,D=8,K=2` timing comparison gives 32.73/32.45 s
for harmonic growth/joint and 34.16/33.10 s for soft Coulomb. Despite operator
factor ranks 2 versus 44, batching limits the time overhead to 2--4%; peak CUDA
memory rises from 19.1 MB to 27.4 MB.

## Novelty and decision

Electronic QMC already contains Pfaffian, multi-Pfaffian, and backflow trial
states, while FermiNet and PauliNet are powerful continuous first-quantized
fermion solvers. FEMPS cannot claim priority for those ingredients. Its candidate
distinction remains the 2201 functional-basis Galerkin layer plus deterministic
exterior/Pfaffian contractions and separate D/K controls.

Phase 10 passes its numerical exit criterion, but the D=10 overlap condition and
remaining seed dependence argue for revising finite-AGP conditioning/canonical
compression before assembling a paper-scale benchmark suite. Larger-N results
remain evidence at measured points only, not an N-scaling claim.

Raw evidence is in `results/soft_coulomb_n4_truth_sweep.json`,
`results/soft_coulomb_n4_k_hierarchy.json`,
`results/soft_coulomb_n4_k4_seed_sweep.json`,
`results/soft_coulomb_n4_d10_k_hierarchy.json`,
`results/soft_coulomb_n6_greedy_k2.json`, and
`results/soft_coulomb_matched_timing.json`.
