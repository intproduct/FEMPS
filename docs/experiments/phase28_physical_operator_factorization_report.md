# Phase 28 physical two-body factorization audit

## Problem

The quadrature-kernel eigendecomposition used by the soft-Coulomb backend is
accurate through the accepted `D=8` benchmark, but projecting tiny kernel modes
into higher harmonic bases amplifies roundoff. Even with all 128 kernel modes,
the dense physical-operator reconstruction errors are `2.15e-10` at `D=10`
and `2.72e-8` at `D=12`, above the registered `1e-11` tolerance.

The tolerance was not relaxed. A general physical operator-Schmidt backend now
groups a dense pair tensor as `(p,r)|(q,s)`, applies an SVD in the `D^2` space,
and returns the explicitly particle-exchange-symmetrized factorization. Its
diagnostics report retained/discarded rank, spectral threshold, dense relative
error, and particle-exchange residual.

## Exact reconstruction checks

Tests cover arbitrary exchange-symmetric real four-index tensors and direct
soft-Coulomb quadrature tensors at `D=10,12`. With relative spectral threshold
`1e-13`, the soft-Coulomb results are:

| `D` | Exterior dimension for `N=4` | Physical SVD rank | Dense relative error |
|---:|---:|---:|---:|
| 8 | 70 | 15 | `1.263e-15` |
| 10 | 210 | 19 | `1.340e-15` |
| 12 | 495 | 23 | `3.847e-15` |

The physical factorization is both more accurate and much lower rank than the
50--128 kernel channels in this basis range. This is an operator-structure
advantage, not a generic FEMPS contraction result.

## D10 continuation pilot

The accepted seed-17 `D=8,K=4` state was zero-padded into `D=10`; no CI
eigenvector was used for initialization. With 80 Adam steps and 40 L-BFGS
refinement steps:

- energy: `11.023237957929902`;
- dense finite-basis CI energy: `11.023133765392014`;
- same-basis error: `1.041925e-4`;
- variance: `9.668484e-4`;
- norm error and both antisymmetry residuals: zero;
- total time: `58.48 s`;
- sampled peak process RSS: `704,540,672` bytes.

This pilot remains numerical evidence. It establishes that the previous
factorization error was a backend-conditioning problem rather than a reason to
weaken the operator tolerance. The next formal point is a checkpointed
`D=10 -> 12` continuation with the direct `D=12` exterior CI audit enabled.
