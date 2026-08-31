# Gate A exact-contraction benchmark

## Outcome

Three independent norm algorithms, two one-body algorithms, and two two-body
algorithms agree in forward values and reverse-mode gradients on deterministic
complex `chi>1` systems. Exterior-coordinate propagation removes exponential
virtual-path enumeration, but its state space still has dimension
`binom(D,p)`. This closes the **generic** route without a PASS. Subsequent
Pfaffian work produced a CONDITIONAL Gate A decision; see
`gate_a_agp_report.md`.

Raw records are stored in:

- `results/gate_a_norm_scaling.json`;
- `results/gate_a_norm_chi3.json`;
- `results/gate_a_norm_chi4.json`.

## Particle-number scaling

CPU, complex128, `D=2N`, `chi=2`, median of two runs:

| `N` | Path-pair determinants | Exterior peak coefficients | Path time (s) | Exterior time (s) | Full coefficients |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 8 | 0.0007 | 0.0009 | 16 |
| 3 | 16 | 30 | 0.0012 | 0.0036 | 216 |
| 4 | 64 | 112 | 0.0040 | 0.0201 | 4,096 |
| 5 | 256 | 420 | 0.0141 | 0.1090 | 100,000 |
| 6 | 1,024 | 1,584 | 0.0545 | 0.4852 | 2,985,984 |
| 7 | 4,096 | 6,006 | 0.2266 | 2.2870 | 105,413,504 |

All available norm routes agree to relative error below `2.2e-15` in this
scan. Full-tensor timings were intentionally stopped after `N=4`.

## Bond-dimension scaling

CPU, complex128, `N=5`, `D=10`, median of two runs:

| `chi` | Paths | Path pairs | Exterior peak coefficients | Path time (s) | Exterior time (s) |
|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 256 | 420 | 0.0141 | 0.1090 |
| 3 | 81 | 6,561 | 630 | 0.3527 | 0.1355 |
| 4 | 256 | 65,536 | 840 | 3.3692 | 0.1694 |

The exterior route exhibits the predicted low-order bond dependence, while
the path route follows the square of `chi^(N-1)`. The largest cross-route
relative difference is `2.5e-14`.

## Interpretation

The exterior dynamic program is already the correct exact replacement for
virtual-path enumeration in small-system algebra work. It is not a scalable
continuous solver: at `D=2N`, its central exterior sector grows like
`binom(2N,N)`, asymptotically `4^N/sqrt(pi N)`.

The measured timings include Python-loop overhead and are not performance
claims. The exact structural counts, rather than a crossover on this small
grid, determine Gate A. GPU optimization cannot change either exponential.

The known polynomial diagonal-path subclass is exactly a finite Slater sum.
It is a valid fallback and sanity check, but does not yet establish a new
polynomially contractible correlated FEMPS family.
