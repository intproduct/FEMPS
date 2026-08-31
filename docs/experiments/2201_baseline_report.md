# Phase 0 report: 2201 functional-MPS baseline

## Outcome

The two-body functional-MPS pipeline is operational. A controlled basis,
bond-dimension, and random-seed scan reaches the energy-error scale reported
in arXiv:2201.12823 for the selected small case.

| Quantity | Value |
|---|---:|
| Oscillators `N` | 4 |
| Functional basis order `D` | 8 |
| MPS bond dimension `chi` | 16 |
| Coupling `gamma` | -0.5 |
| Three-body coupling | 0 |
| Optimizer | Adam + cosine LR, 1500 steps |
| Dtype | complex128 |
| Device | NVIDIA RTX PRO 4000 Blackwell (`cuda:2`, sm_120) |
| Exact continuum energy | 1.8786948647835056 |
| Final variational energy | 1.878706464313651 |
| Absolute error | 1.159953014551185e-5 |
| GPU wall time | 40.49 s |

The original single-point record is stored in
`results/2201_n4_d8_chi16_gpu_cosine.json`. The controlled scan and every
per-point convergence trace are stored under `results/2201_sweep/`.

## Controlled convergence scan

All points use `N=4`, `gamma=-0.5`, complex128, 1500 Adam steps with a cosine
learning-rate schedule, and the Blackwell GPU. The exact continuum energy is
`1.8786948647835056`.

### Functional-basis truncation (`chi=16`, seed 0)

| `D` | Parameters | Final energy | Absolute error |
|---:|---:|---:|---:|
| 2 | 40 | 1.9058031623 | 2.711e-2 |
| 4 | 544 | 1.8798273483 | 1.132e-3 |
| 6 | 1,224 | 1.8787441452 | 4.928e-5 |
| 8 | 2,176 | 1.8787064643 | 1.160e-5 |
| 10 | 3,400 | 1.8787038785 | 9.014e-6 |
| 12 | 4,896 | 1.8787057533 | 1.089e-5 |

### MPS bond truncation (`D=8`, seed 0)

| `chi` | Parameters | Final energy | Absolute error |
|---:|---:|---:|---:|
| 2 | 96 | 1.8867672862 | 8.072e-3 |
| 4 | 320 | 1.8788107786 | 1.159e-4 |
| 8 | 1,152 | 1.8787181288 | 2.326e-5 |
| 12 | 1,664 | 1.8787087682 | 1.390e-5 |
| 16 | 2,176 | 1.8787064643 | 1.160e-5 |
| 20 | 2,688 | 1.8787038688 | 9.004e-6 |

### Initialization stability (`D=8`, `chi=16`)

Across seeds 0--3, the absolute-error mean is `1.179e-5`, the population
standard deviation is `1.428e-6`, and the range is
`[1.046e-5, 1.416e-5]`. The full 14-point scan took 566.31 s.

## Validation evidence

- 10 CPU/integration tests pass.
- Native latticeTN contractions equal a dense Rayleigh quotient at `1e-11`
  tolerance for a small random functional MPS.
- CPU/Blackwell forward energy differs by `1.78e-15`.
- CPU/Blackwell parameter gradients differ by at most `8.69e-16`.
- The final variational energy remains above the exact continuum ground energy.

## Interpretation and limits

This reproduces the functional-basis-to-MPS-to-AD mechanism and the qualitative
`D`/`chi` convergence of the paper without claiming a pixel-level digitization
of Figs. 3 and 4. The `D=10` to `D=12` non-monotonicity and the multi-seed spread
show that the final `1e-5` plateau contains finite-optimization error; it should
not be interpreted as a pure basis error. Every result remains variational.

The exercise also exposed two migration details:

1. the rendered derivative Eq. (9) has an inconsistent first ladder
   coefficient, so the implementation uses the exact ladder identity;
2. upstream latticeTN's default MPS geometry assumes local dimension two, so
   FEMPS creates explicit `D`-dimensional cores while reusing its contractions.
