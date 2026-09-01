# ADR 0008: Basis-efficiency and N=8 Gate F passes with an auxiliary qualification

- Status: accepted
- Date: 2026-09-01

## Context

Gate E admitted an unbounded ordered-distance functional TN through N=6, but
the N=6 error was basis dominated and the production builder temporarily
materialized dense raw Fourier MPO bulk tensors. It did not admit N=8.

Phase 18 introduces a Lowdin-orthonormalized two-scale odd-Hermite half-line
basis. Analytic Gaussian-polynomial overlap and local-operator matrices agree
with independent quadrature, and all primitives obey the collision Dirichlet
boundary. At matched N=4 orders the multiscale errors are 40--70% smaller than
single-scale odd Hermite. At N=6 the D=10 error against the exterior D=12
numerical reference is `2.452e-3`, an `81.5%` reduction from Gate E's D=8
single-scale error.

The Fourier recurrence is now contracted into a retained left transfer and
compressed one site at a time. Small global operators match raw-then-compress
to `1.6e-15` relative Frobenius error. At N=8,D=10, the largest construction
intermediate has 5,465,600 entries instead of the theoretical raw 109,482,800,
and no dense raw Fourier bulk is materialized.

Three blind N=8,D=10 global-AD seeds give energies
`44.4543733--44.4544088`. The best differs by `8.364e-3` from an exterior D=12
numerical reference, below the declared `1.2e-2` budget. An independent
chi-16 local DMRG audit gives `44.4543222`, within `5.11e-5` of the source
chi-32 AD state. Chi-32 DMRG itself is resource rejected because the current
local contraction requests a 78.12 GiB intermediate on a 23.89 GiB GPU.

MPO bonds 128 and 192 differ by `3.60e-7` in fixed-state energy, passing the
declared `1e-6` budget. An additional raw parameter-gradient relative threshold
of `2e-6` misses: the observed difference is `2.90e-5`, although the gradient
cosine similarity is `0.999999999584`. This auxiliary failure is retained.

## Decision

Accept Gate F as a controlled N=8 ordered-distance functional-TN point, with
the raw-gradient and chi-32 local-optimizer qualifications explicitly carried
forward.

Acceptance conditions are:

1. keep exterior D=12/D=14 values labeled numerical references, not continuum
   bounds;
2. retain basis order, central scale, scale ratio, Fourier order/cutoff, local
   quadrature, MPO bond, MPS bond, and optimization schedule as independent
   controls;
3. use the incremental structured MPO path for production whenever compression
   is requested, and globally audit new bond choices;
4. do not interpret local discarded MPO singular values as global
   certificates;
5. report the bond-128 gradient auxiliary miss and do not present bond 128 as
   gradient-converged to the stricter `2e-6` threshold;
6. confine dense product-basis vectors to bounded post-training truth audits;
7. treat chi-32 local DMRG as unadmitted until the effective-Hamiltonian
   contraction order is repaired and resource audited; and
8. continue to call the route ordered-distance functional TN, not FEMPS.

## Consequences

- N=8 is admitted as one controlled numerical point, not as an asymptotic
  scaling result.
- The Gate E dense raw-MPO construction bottleneck is closed.
- The two-scale basis becomes the default larger-N control, while single-scale
  odd Hermite remains a matched comparator.
- N=10 is not admitted. Phase 19 must address local-optimizer intermediates,
  the auxiliary MPO-gradient convergence miss, and stronger N=8 basis/
  reference controls before any larger-particle claim.
- The novelty boundary is unchanged: Hong et al. and Li--Waintal remain the
  direct method parents, and this is an integration/evidence result.

