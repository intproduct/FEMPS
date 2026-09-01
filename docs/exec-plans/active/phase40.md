# Active execution plan: Phase 40 N=2 Explicit-Correlation Differentiator Gate

## Objective

Determine, before any second-manuscript writing, whether a symmetric explicit
correlator multiplying a first-quantized continuous exterior carrier produces
a reproducible functional-basis `D`-convergence advantage that optimized
fixed-`K` NOCI does not reproduce.

This is an algorithm experiment, not Paper B. No title, abstract, outline, or
submission source for a second paper may be created during this phase.

## Frozen physical problem

- two spinless fermions in one spatial dimension;
- unit-frequency harmonic trap;
- soft-Coulomb interaction with coupling `1` and softening `1`;
- unit-frequency harmonic functional basis;
- independent relative-coordinate grid reference;
- deterministic float64 calculations and recorded seeds.

## Frozen state families and axes

1. Correlated exterior carrier
   `exp(sum_m a_m exp(-beta_m (x1-x2)^2)) Psi_A`, with symmetric pair features.
   Vary carrier basis `D` and feature count `P` independently. Use nested
   exponent sets `P=0`, `P=1: {1}`, `P=3: {1/4,1,4}`, and
   `P=5: {1/16,1/4,1,4,16}`.
2. Optimized NOCI controls with fixed `K in {1,2,4}` at each admitted `D`.
3. Same-basis full CI and the independent relative-coordinate reference.
4. The existing ordered-coordinate result may be reported as a named
   Li--Waintal-style control when its basis and cost are disclosed; it is not
   FEMPS.

The production `D` axis is `D in {2,4,6,8,10,12}`. Carrier orbitals and
correlator amplitudes must be optimized from frozen initializations and
budgets. Parameter counts, wall time, and peak RSS are reported for every
family. The correlator state generally lies outside finite `Lambda^2 V_D`, so
same-basis CI is a comparator rather than an equal variational space; every
table must state this.

## Validation before production

- [ ] Materialized state/value equivalence at the smallest `D/P/K` points.
- [ ] AD versus central finite differences for both carrier and correlator
  parameters.
- [ ] Antisymmetry residual at or below `1e-12` for every reported state.
- [ ] `Q=128` to `Q=160` energy change at or below `1e-7` and relative norm
  change at or below `1e-8`; otherwise increase the frozen quadrature audit
  uniformly and record the preregistration amendment before production.
- [ ] Independent reconstruction of serialized observables.

## Reported quantities

For every admitted point report energy and reference error, raw norm and norm
quadrature error, energy variance, antisymmetry residual, gradient-check error,
optimizer outcome, parameter count, wall time, peak RSS, and whether any
`Q^2`, `D^N`, full alternating coefficient tensor, or virtual-path object was
materialized. `D` and `P`/`K` convergence must appear as separate axes.

## Differentiator gate

Passing requires all validation checks and a reproducible advantage over the
optimized fixed-`K` NOCI controls. The report must show, at more than one
consecutive `D`, either a smaller reference error at a disclosed matched
parameter/cost envelope or a clearly better error-versus-`D` trend that is not
explained by increasing `K`. The comparison must include uncertainty and
optimizer stability; a single favorable point is insufficient.

Projected Slater rank growth is supporting numerical evidence only. Lower
energy than same-basis CI is not by itself a pass because the explicitly
correlated state occupies a larger coordinate-function space.

## Failure and publication rule

If the differentiator is absent, unstable, or dependent on unmatched cost,
reject this route and report the negative result. Do not tune thresholds after
seeing production outcomes. Whether the gate passes or fails, Phase 40 does
not create a second manuscript. Only after an independently reproduced gate
pass may the project make a new publication decision.
