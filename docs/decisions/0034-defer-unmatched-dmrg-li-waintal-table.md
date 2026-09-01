# ADR 0034: Defer an unmatched DMRG or Li--Waintal table

## Status

Accepted on 2026-09-02 after the failed Phase 44 gate and before any new
comparator calculation.

## Context

Phase 44 supplies internally verified low-`D` explicit-correlation evidence,
but its complete preregistered gate fails the blind-selection SE/ESS contract.
The project permits only a genuinely matched Li--Waintal or same-basis DMRG
comparison as the alternative method-paper differentiator. Adding a method
name or an unmatched energy column is not sufficient.

At the Phase 44 sizes, the exterior dimensions are only 1, 15, and 70 for
`D=4,6,8`. Direct CI already diagonalizes the exact projected Hamiltonian and
provides stronger energy truth than a truncated second-quantized DMRG solve.
DMRG would be scientifically useful only outside this exact-CI region under a
separately controlled truncation/bond schedule.

The existing ordered-coordinate implementation is a named continuation of
Li--Waintal, but its controlled N4 point uses COM/positive-gap coordinates,
a half-line sine basis with `D=10`, box `Rmax=4.5`, interaction polynomial
degree 20, MPS bond 32, and 6,600 parameters. Its dominant basis/box error is
about `4.35e-3`. Phase 44 instead uses full-line harmonic carrier orders
`D=4,6,8`, a five-feature explicit correlator, coordinate VMC uncertainty, and
37 raw parameters at D8. The meanings of `D`, state space, truncation error,
and cost are not matched.

## Decision

Do not run an immediate same-basis DMRG or Li--Waintal comparison table.

1. Retain exact exterior CI as the stronger current same-basis comparator.
2. Do not enlarge N or D merely to make DMRG nontrivial.
3. Do not compare Phase 44 and the existing ordered-coordinate point as if
   their basis order or parameter count were equivalent.
4. Do not append same-point FEMPS samples to repair Phase 44.
5. Keep Paper B closed.

A future comparator is admissible only under a new ADR that fixes a common
physical model, an explicit accuracy target, independently converged basis/
domain/operator errors, matched time and peak-memory accounting, all bond or
sampling axes, software provenance, and the exact claim to be tested. The
comparison must answer something not already supplied by direct CI.

## Consequence

Phase 45 closes as a negative feasibility decision rather than producing an
unmatched table. Immediate work moves to external human review of the algebraic
complexity claims and external reproduction handoff for the Phase 44 data.
No new paper or numerical gate is opened by this ADR.
