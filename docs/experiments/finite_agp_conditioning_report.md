# Phase 11 finite-AGP conditioning report

## Scope and correction to the Phase 10 diagnosis

This phase studies the fixed-number finite-AGP subclass admitted by the
CONDITIONAL Gate A decision.  It does not establish polynomial contraction for
generic matrix-wedge FEMPS.

The Phase 10 `D=10,K=4` overlap condition number of `143.5` was computed from
the raw overlap matrix.  That number is not invariant under independent scale
and phase choices for the nonlinear AGP terms.  Direct diagnosis shows that the
largest normalized inter-term state overlap is only `0.186`; the raw condition
is caused primarily by unequal term norms, not near-linear dependence.

For overlap `S`, define `D_aa=sqrt(S_aa)` and the unit-diagonal correlation
matrix

`C = D^{-1} S D^{-1}`.

If `C=U Lambda U^dagger`, the retained whitener is

`W = D^{-1} U_r Lambda_r^{-1/2}`,

so `W^dagger S W=I`.  Compression thresholds now act on `C`, making the rank
decision invariant under independent nonzero term rescalings and phases.  The
linear combinations produced by `W` are coordinates in the current AGP span;
they are not claimed to be individual AGPs.

| Checkpoint | Raw condition | Balanced condition | Max normalized overlap |
|---|---:|---:|---:|
| D=8, K=4 | 2.081 | 1.359 | 0.137 |
| D=10, K=4 | 143.513 | 1.464 | 0.186 |

## Gauge-invariant correlation diagnostic and safe pruning

For optimized amplitudes `c`, the contribution Gram matrix is

`G_ab = c_a^* S_ab c_b / (c^dagger S c)`.

Its spectrum is invariant under independent term scale/phase gauges and term
permutations.  It measures multiplicity of contributions in a finite AGP sum;
it is explicitly **not** a particle entanglement spectrum.

The pruning rule never uses the raw condition number.  It proposes removal
only when the balanced overlap loses rank or exceeds a chosen condition
threshold, and only if a full leave-one-term-out amplitude solve raises the
energy by no more than an explicit tolerance.  A duplicate-term regression
test verifies unit fidelity of the explicit exterior state after the proposed
deletion.  No production run in this phase triggered a deletion or restart.

## Conditioned D=10, K=4 three-seed refinement

Three independent blind `D=8,K=4` chains from Phase 10 were embedded in the
first eight orbitals of `D=10` and optimized at `Q=128`.  No truth state entered
the initialization or optimization.

| Seed | Finite-basis error | Fidelity | Balanced condition | Raw condition | Discarded |
|---:|---:|---:|---:|---:|---:|
| 301 | 1.334e-5 | 0.99999870 | 1.750 | 3.143 | 0 |
| 302 | 1.900e-5 | 0.99999833 | 3.062 | 3.301 | 0 |
| 303 | 2.031e-5 | 0.99999820 | 2.340 | 3.469 | 0 |

All three reproduce or improve the Phase 10 `D=10,K=4` error `2.040e-5`.
The polynomial/exterior energy mismatch is at most `2.06e-13`.  The K4 error
spread is `6.97e-6`.

## Reproducible K=5 improvement

One randomly seeded fifth term was grown against each frozen K4 prefix, then
all five terms were released for a shorter joint stage.

| Seed | K4 error | K5 error | Improvement | Balanced condition | Discarded |
|---:|---:|---:|---:|---:|---:|
| 301 | 1.334e-5 | 6.000e-6 | 7.339e-6 | 1.751 | 0 |
| 302 | 1.900e-5 | 8.820e-6 | 1.018e-5 | 3.102 | 0 |
| 303 | 2.031e-5 | 8.026e-6 | 1.228e-5 | 2.518 | 0 |

K5 improves every chain.  The mean error falls from `1.755e-5` to `7.615e-6`
and the spread falls from `6.97e-6` to `2.82e-6`.  Polynomial and explicit
exterior energies agree within `8.35e-14`; all final balanced conditions remain
below `3.11`.

## Decision

Phase 11 passes.  The apparent D10 pathology was a gauge-dependent norm
imbalance, and diagonal balancing removes it without changing the variational
span.  The conditioned finite-AGP subclass is ready for a broader controlled
benchmark matrix in `(N,D,K)`.  This is readiness of the admitted structured
subclass, not evidence that generic FEMPS has passed Gate A, and the
contribution spectrum is not promoted to an entanglement measure.

Raw evidence is in
`results/soft_coulomb_conditioning.json`,
`results/soft_coulomb_conditioned_d10_k4_seeds.json`, and
`results/soft_coulomb_conditioned_d10_k5_seeds.json`.
