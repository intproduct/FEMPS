# ADR 0004: Ordered-sector functional TN becomes the primary numerical route

- Status: accepted
- Date: 2026-09-01

## Context

ADR 0003 conditionally rejects unrestricted dense matrix-wedge FEMPS exact
contraction. Phase 14 then classified the principal restricted coefficient
algebras. The proved tractable cases reduce to AGP/LC-AGP or Gaussian
structure; bounded nilpotent/triangular hierarchies put the improvement order
in the polynomial exponent and overlap AGP jets/border rank; a noncommutative
semisimple `Mat_2` block contains the hard reduction.

The ordered-coordinate representation is unitarily equivalent to the
antisymmetric full-space problem, with collision boundaries carrying Pauli
exclusion. A deterministic `N=4,D=8` soft-Coulomb comparison gives ordered
particle-MPS ranks `(5,9,5)` versus full antisymmetric ranks `(8,28,8)`, while
preserving exact signed reconstruction. The ordered tensor has an exact
latticeTN MPS representation, native norm contraction, and differentiable
truth energy.

## Decision

Make ordered-sector/interparticle-distance functional TN the primary numerical
research route after the generic FEMPS gate failure.

This route:

1. remains first-quantized and coordinate-functional;
2. reuses 2201/latticeTN MPS, MPO, AD, checkpoint, and device infrastructure;
3. carries exclusion through the ordered domain and collision boundary;
4. must develop a scalable constraint/operator representation before claiming
   solver success; and
5. is not called FEMPS or presented as a new exterior ansatz.

Retain matrix-wedge FEMPS as a mathematical ansatz and contraction-obstruction
result. Retain finite LC-AGP as a controlled functional-basis baseline only.

## Consequences

- The next contraction gate concerns distance-coordinate kinetic, trap,
  interaction, and finite-box constraint MPOs.
- Dense ordered gathering remains a truth oracle, not production code.
- Li--Waintal first-quantized MPS is the closest prior method and sets the
  priority boundary; the possible contribution is the 2201 continuous
  functional-basis/operator/AD integration and associated no-go analysis.
- If the ordered operators do not admit controlled polynomial MPOs, the
  numerical solver branch stops and the project continues as theory/no-go.
