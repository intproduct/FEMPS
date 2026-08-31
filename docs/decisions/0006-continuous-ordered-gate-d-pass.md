# ADR 0006: Continuous ordered-distance Gate D passes at controlled scope

- Status: accepted
- Date: 2026-09-01

## Context

Gate C admitted an exact finite-grid ordered-distance MPS/MPO but did not
recover the orthonormal continuous functional-basis structure of Hong et al.
The open requirements were an exact continuous coordinate calculus, collision
boundary, functional operator matrices, a native interaction MPO, blind AD,
and independent controls for basis/scale, interaction separation, and MPS bond.

Phase 16 uses center of mass plus positive interparticle gaps. The map has unit
Jacobian, exact kinetic and harmonic metrics, `sqrt(N!)` chamber normalization,
and exact signed reconstruction. A Dirichlet sine basis supports a controlled
finite interval and a polynomial soft-Coulomb MPO; an odd-Hermite half-line
basis removes the outer boundary for noninteracting tests. All production
energies and gradients use native latticeTN MPS/MPO contraction.

At N=2, basis and box scans converge to an independent half-line reference and
all three blind runs are within `3.01e-6` of their post-run Galerkin truth. At
N=4, independent MPO/Lanczos audits separate the finite-basis, interaction,
MPS-rank, and optimizer errors. Three blind Blackwell runs lie
`4.37e-3--4.39e-3` above an exterior `D=14` numerical reference, below the
predeclared `6e-3` tolerance. CPU/GPU energy and gradients agree to float64
precision.

## Decision

Accept Gate D as a controlled continuous-basis prototype and proceed to
unbounded-interaction and scaling work.

The acceptance has the following conditions:

1. call the method ordered-distance functional TN, not FEMPS;
2. attach no priority claim to the ordered chamber or distance-variable MPS;
3. treat the exterior `D=14` energy as a numerical reference, not a continuum
   bound;
4. use the exact raw interaction MPO unless any compression has an independent
   global operator/action audit;
5. report `D`, box/scale, interaction degree, MPS bond, and optimization as
   separate controls;
6. do not apply the finite-interval soft-Coulomb polynomial to an unbounded
   basis; and
7. confine dense product-basis states to explicitly bounded truth audits.

## Consequences

- The 2201 continuous functional-basis/operator/AD machinery is now validated
  for the ordered fermionic route through N=4.
- The dominant N=4 production error is the finite sine basis. Improving the
  local half-line basis is higher priority than increasing MPS bond.
- Odd Hermite is admitted as a promising noninteracting basis candidate, but
  needs a new separable interaction representation before production use.
- Larger-N work must measure accuracy-to-`D`, accuracy-to-`chi`, raw/compressed
  MPO scaling, and optimizer stability before any scalability statement.
- Generic matrix-wedge FEMPS remains conditionally obstructed; this ADR does
  not change Gate A or create an exterior-carrier method claim.
