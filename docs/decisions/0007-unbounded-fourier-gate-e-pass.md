# ADR 0007: Unbounded Fourier--Bessel Gate E passes at controlled N=6 scope

- Status: accepted
- Date: 2026-09-01

## Context

Gate D found that the finite sine basis and outer box dominated the N=4 error.
It left the interacting odd-Hermite half-line basis, safe MPO compression, and
N greater than four unresolved.

Phase 17 represents soft Coulomb by a truncated Fourier--Bessel cosine rule.
Projected cosine/sine multiplication matrices act on the odd-Hermite basis.
A four-state real recurrence per Fourier node accumulates every particle-pair
cosine, reducing the raw interaction bond from direct-pair growth to `4M`,
independent of particle count. Direct-pair equality, direct half-line
quadrature, Fourier/local-quadrature convergence, and global compression
errors are measured independently.

At matched N=2 and N=4 basis orders, odd Hermite improves the finite sine box
at every tested point. At N=6,D=8, blind global AD on RTX PRO 4000 Blackwell
finishes within `2.49e-4` of its post-run Galerkin truth for all three seeds.
The training MPO compression has `1.63e-9` relative global action error, and
the total difference from an exterior D=12 numerical reference is `1.328e-2`,
below the declared `2e-2` controlled-point tolerance. TT-SVD shows that MPS
bond eight already has only `4.50e-6` representation error.

## Decision

Accept Gate E as a controlled unbounded interacting prototype through N=6 and
proceed to basis-efficiency and larger-particle work.

The acceptance has the following conditions:

1. call the method ordered-distance functional TN, not FEMPS;
2. attach no priority claim to ordered chambers, distance variables, or
   first-quantized MPS;
3. retain independent controls for Fourier order, frequency cutoff, local
   projection quadrature, functional basis order/scale, MPO bond, MPS bond,
   and optimization;
4. certify every production MPO compression by a bounded global operator or
   action audit, never by local discarded singular values alone;
5. treat exterior D=12/D=14 energies as numerical references, not continuum
   bounds;
6. confine product-basis state materialization to post-training bounded truth
   audits; and
7. do not infer N=8 or asymptotic continuum scaling from the N=6 pass.

## Consequences

- The interacting odd-Hermite route supersedes the finite sine box as the
  default controlled continuous ordered solver.
- The compact interaction bond is independent of N at fixed Fourier order,
  but the present dense raw-MPO tensors still create an avoidable memory cost.
- The dominant N=6 discrepancy is basis order, not MPS capacity or optimizer
  accuracy. Improving half-line basis efficiency takes priority over larger
  `chi`.
- N=8 admission requires a new controlled basis/error budget and resource
  record; it is not granted by this ADR.
- The novelty boundary from ADR 0006 is unchanged. This is an integration and
  evidence contribution between Hong et al. and Li--Waintal, not a new
  ordered-sector ansatz claim.
