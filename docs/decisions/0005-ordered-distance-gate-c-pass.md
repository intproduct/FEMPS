# ADR 0005: Ordered-distance Gate C passes at finite-grid scope

- Status: accepted
- Date: 2026-09-01

## Context

ADR 0004 selected the ordered-sector/interparticle-distance route only
conditionally.  The Phase 14 comparator still gathered a dense particle tensor
and had no production representation for the kinetic operator, finite-box
constraint, or soft-Coulomb interaction.

Phase 15 replaces that path by `N+1` nonnegative empty-site gaps with exact
total charge `G=L-N`.  Adjacent gap transfers reproduce finite-difference
kinetic hopping; cumulative gap number operators reproduce the harmonic trap;
and interval counters reproduce every soft-Coulomb particle pair.  A hard-charge
MPS enforces the box constraint without a penalty.  The resulting raw MPO has
an explicit `O(N^2 G)` bond bound, and latticeTN evaluates and differentiates
the Rayleigh quotient without a local-dimension-to-the-number-of-sites gather.

On the controlled `N=4,L=8` problem, exact native contraction agrees with the
ordered truth to `5.33e-15`.  Three random Blackwell optimizations have energy
errors between `1.30e-6` and `2.07e-5`, exact charge weight one, and fidelity
above `0.9999956`.

## Decision

Accept Gate C for the finite-grid ordered-distance representation and continue
the numerical branch to a continuous functional-basis gate.

The acceptance has the following conditions:

1. production energy and training use the exact raw MPO unless an approximate
   MPO has a separately reported operator-error control;
2. optimization uses the hard cumulative-charge sector rather than an
   uncontrolled finite-box penalty;
3. dense gap tensors and exact diagonalization remain small truth audits only;
4. every result separates grid/box, local gap cutoff, MPS bond, and MPO error;
5. the method is called ordered-distance functional TN, not FEMPS; and
6. no continuum or large-particle scaling claim is inferred from this gate.

## Consequences

- Phase 16 will introduce full-line center-of-mass and Dirichlet half-line gap
  functional bases, derive their differential and cumulative-interaction
  operators, and test them first against exact small systems.
- The closest prior art remains Li--Waintal's ordered first-quantized MPS.  The
  possible contribution is restricted to the 2201 functional-basis/operator/AD
  integration, controlled continuum evidence, and the accompanying no-go
  theory.
- Matrix-wedge FEMPS remains a mathematical/no-go object; finite LC-AGP remains
  a benchmark baseline.
- Failure of the continuous-basis or scaling gate will stop solver expansion
  without revoking the finite-grid representation result.
