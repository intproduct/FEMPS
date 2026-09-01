# ADR 0030: Preregister the N=2 explicit-correlation differentiator gate

## Status

Accepted on 2026-09-02 before Phase 40 production runs.

## Context

The Phase 39 bounded prototype shows that a symmetric Jastrow multiplier can
preserve exact antisymmetry and generate increasing finite-basis projection
rank from a single exterior Slater carrier. It does not show an advantage over
optimized fixed-`K` NOCI. The exponent choices and numerical tolerances below
are informed by that exploratory pilot, so they are frozen now and must not be
described as independently predicted.

## Decision

Run one `N=2` soft-Coulomb experiment with the physical model, nested `D/P/K`
axes, controls, validation tolerances, reported quantities, and failure rule in
`docs/exec-plans/active/phase40.md`. Freeze initialization and optimizer seeds
in the experiment configuration before the first production result is
generated. Do not add a backup ansatz during this phase.

The primary scientific comparison is explicit-correlation error versus
functional basis `D`, against optimized fixed-`K` NOCI with parameter and cost
disclosure. Same-basis CI is a finite-basis control, and the independent
relative-coordinate solution supplies the energy reference. Li--Waintal-style
ordered coordinates remain a named optional comparator rather than FEMPS.

## Publication consequence

This ADR authorizes algorithm development and benchmark execution only. It
does not authorize a second manuscript. A separate publication decision may
be considered after, and only after, an independent reproduction confirms the
Phase 40 differentiator gate. Until then all solver numerics remain candidate
material for the one combined structural/no-go manuscript.
