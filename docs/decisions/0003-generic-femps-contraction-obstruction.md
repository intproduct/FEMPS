# ADR 0003: Generic exact matrix-wedge FEMPS contraction is obstructed

- Status: accepted
- Date: 2026-09-01

## Context

The project requires norm and Hamiltonian contractions polynomial jointly in
particle number, functional-basis size, and virtual bond. Parameter count alone
is insufficient. Scalar AGP and finite LC-AGP are contractible, but direct
prior art prevents using LC-AGP itself as the central FEMPS novelty.

Phase 13 introduced a virtual-matrix-valued pair power, derived its N=4
contractions, and found an exact polynomial-size tagging reduction from the
row-ordered Cayley determinant over `2 x 2` matrix entries. That determinant is
permanent-hard. The pair-power state has `D=N=2n`, pair bond `O(n)`, and an
explicit embedding into the original one-form matrix-wedge FEMPS with maximum
bond `O(n^2)`. A direct-sum interference argument transfers amplitude hardness
to exact norm evaluation.

## Decision

Treat unrestricted dense matrix-wedge FEMPS exact contraction as Gate FAIL,
conditional on the standard permanent-complexity assumption. Do not develop a
generic GPU solver or advertise polynomial exact contraction for this family.

Retain:

1. the ansatz and reduction as a mathematical/no-go result;
2. scalar and finite LC-AGP as validated controls and fallback calculations,
   not as the core novelty;
3. exact small-system exterior code as theorem and regression oracles; and
4. the gauge-invariant one-body correlation multiplicity as a diagnostic only.

Further solver work requires one of:

1. a physically meaningful restricted core algebra with a proved joint
   polynomial contraction and systematic improvement path;
2. a stronger statistics-carrier times correlation-multiplicity factorization;
3. an ordered-sector first-quantized functional TN; or
4. an explicitly approximate contraction with controlled error and exact
   antisymmetry.

## Consequences

- Generic matrix-wedge benchmark expansion stops at exact small systems.
- Any restricted subclass must be audited against LC-AGP/border-rank,
  Gaussian fermionic, symmetry-adapted TN, and noncommutative determinant
  literature before implementation.
- The main scientific narrative becomes no-go plus restricted/alternative
  representation analysis unless a later subclass passes a new gate.
