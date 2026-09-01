# Phase 28 nonquadratic soft-Coulomb transferability report

## Scope and boundary

This report is **numerical evidence** for the restricted nonbranching
diagonal-path FEMPS under the nonquadratic continuous interaction

\[
  H=\sum_i\left(-\frac12\partial_{x_i}^2+\frac12x_i^2\right)
  +\sum_{i<j}\frac{1}{\sqrt{(x_i-x_j)^2+1}},\qquad N=4.
\]

The state remains first quantized and uses continuous harmonic-oscillator
functional bases. It is a nonorthogonal multideterminant FEMPS subclass, not a
generic FEMPS contraction and not a second-quantized MPS. The benchmark and
independent verifier are

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_phase28_soft_coulomb_transferability.py
.\.venv\Scripts\python.exe scripts\verify_phase28_soft_coulomb_transferability.py
```

Raw data are in
`results/phase28_soft_coulomb_transferability.json`. Checkpoints are ignored
reproduction artifacts. No CI eigenvector initializes a FEMPS run.

## Registered acceptance

Before the formal run, the following thresholds were fixed:

- dense-quadrature finite-basis CI error at most `2e-4`;
- energy variance at most `2e-3`;
- norm error at most `1e-10`;
- structural and materialized antisymmetry residuals at most `1e-12`;
- factorized soft-Coulomb operator error at most `1e-11`;
- three blind `D=6,K=4` seeds and three truth-free `D=6 -> 8`
  continuations must all pass;
- energy must be nonincreasing on the independent `K=1,2,4` axis;
- error relative to the pre-existing `D=14` numerical reference must decrease
  from `D=6` to `D=8` at fixed `K=4`;
- every point must record process RSS and enumerate zero virtual paths.

The first formal run rejected the `D=8` operator factorization: its error was
`5.046e-11`, although every state criterion passed. The acceptance threshold
was not relaxed. The operator builder was changed to lower its spectral
truncation deterministically until the registered dense reconstruction error
was met. `D=6` retained threshold `1e-13` and rank 46; `D=8` selected
threshold `1e-14`, rank 50, and error `6.543e-13`. Repeating the full run changed
the first `D=8` state error by only about `2.4e-12`.

## Stability and truth audit

The independent dense truth uses the direct four-index `Q=128` quadrature
tensor and Slater--Condon exterior Hamiltonian, not the production factorized
operator.

| Group | Initialization | Passes | Maximum dense-CI error | Maximum variance | Energy spread |
|---|---|---:|---:|---:|---:|
| `D=6,K=4` | three blind seeded starts | 3/3 | `4.528e-6` | `2.700e-5` | `4.465e-6` |
| `D=8,K=4` | three nested `D=6 -> 8` continuations | 3/3 | `1.509e-4` | `1.128e-3` | `8.010e-5` |

The `D=6` dense-CI energy is `11.023837713203346`; its exterior dimension is
15 and ordinary particle-TT ranks are `(6,15,6)`. The `D=8` dense-CI energy is
`11.023278984749750`; its exterior dimension is 70 and particle-TT ranks are
`(8,28,8)`. Across every FEMPS point, the maximum norm error is
`2.220e-16`, polynomial/materialized energy disagreement is below
`8.882e-15`, and both antisymmetry residuals are zero.

The `D=8` result passes but is not saturated: its state error and variance are
much larger than at `D=6`. This is recorded as a correlation/optimization
limitation at fixed `K=4`, not hidden by quoting only the total basis error.

## Independent correlation and basis axes

At fixed `D=6`:

| `K=chi` | Energy | Dense-CI error | Variance |
|---:|---:|---:|---:|
| 1 | 11.025279022118180 | `1.441e-3` | `6.360e-3` |
| 2 | 11.023866678350620 | `2.897e-5` | `2.009e-4` |
| 4 | 11.023837776253766 | `6.305e-8` | `3.147e-7` |

Thus increasing `K` systematically recovers non-Slater correlation for this
nonquadratic model. At fixed `K=4`, using seed 17:

| `D` | FEMPS energy | Absolute error versus `D=14` numerical reference |
|---:|---:|---:|
| 6 | 11.023837776253766 | `7.549e-4` |
| 8 | 11.023424901566514 | `3.420e-4` |

The `D=14` value `11.023082853674637` is a finite-basis numerical reference,
not a continuum bound. The exact finite-basis errors versus that value are
`7.549e-4` at `D=6` and `1.961e-4` at `D=8`; the larger FEMPS `D=8` value
contains an additional `1.459e-4` correlation/optimization error.

## Cost and measured tradeoff

The `D=6,K=4` runs take `81.4--83.5 s`; the `D=8,K=4` continuations take
`116.6--117.2 s`. Sampled process peak RSS grows from about 768 MB on the first
run to 1.02 GB late in the sequential process, while per-call RSS increases
range from about 59 MB to 217 MB. Absolute RSS includes the loaded Torch
runtime and allocator cache and is not an isolated-process lower bound.

Direct finite-basis CI takes about `0.12 s` at `D=6` and `0.85 s` at `D=8`.
Therefore this controlled region shows no speed, memory, or parameter-count
advantage for FEMPS. The demonstrated FEMPS-specific property is narrower:
exchange symmetry stays exact independently of `K`, while `K` controls a
systematic correlation expansion and production avoids materializing the
particle tensor or enumerating virtual paths. The high soft-Coulomb factor
rank exposes the expected `K^2 L` cost and is a clear adverse tradeoff relative
to the low-rank harmonic interaction.

## Decision

The nonquadratic transferability test passes its registered criteria and the
independent verifier returns `verified: true`. This strengthens the claim that
the restricted diagonal-path FEMPS is a functioning continuous interacting
solver, but does not support generic scalability or superiority claims.

The next bounded step is a single-lineage `D=8 -> 10 -> 12` continuation with
dense-CI audits still enabled. It should determine whether basis convergence
continues while fixed `K=4` correlation error and `K^2 L` cost remain
controlled. Larger `N` or truth-free dimensions should not be entered before
that audit.
