# Active execution plan: Phase 40 N=2 Explicit-Correlation Differentiator Gate

## Status

Resumed on 2026-09-02 after the fixed Phase 39 point and the internal
manuscript-A theory/proof/citation/hygiene closure were completed. External
human review of the algebraic-complexity results remains a separate submission
gate and is not replaced by this algorithm experiment.

This is the only admitted post-A numerical route. It does not add ordinary
small NOCI evidence: NOCI points are generated solely as preregistered controls
for the explicit-correlation differentiator.

## Objective

Determine, before any second-manuscript writing, whether a symmetric explicit
correlator multiplying a first-quantized continuous exterior carrier produces
a reproducible functional-basis `D`-convergence advantage that optimized
fixed-`K` NOCI does not reproduce.

This is an algorithm experiment, not Paper B. No title, abstract, outline, or
submission source for a second paper may be created during this phase.

## Frozen physical problem and axes

- `N=2` spinless fermions in a unit-frequency one-dimensional harmonic trap;
- soft-Coulomb coupling `1` and softening `1`;
- unit-frequency harmonic functional basis and `D in {2,4,6,8,10,12}`;
- symmetric Gaussian correlator features
  `P=0`, `P=1: {1}`, `P=3: {1/4,1,4}` and
  `P=5: {1/16,1/4,1,4,16}`;
- optimized NOCI controls `K in {1,2,4}`;
- same-basis CI and an independent relative-coordinate grid reference.

The explicitly correlated state generally lies outside finite
`Lambda^2 V_D`; same-basis CI is therefore a comparator, not an equal-space
variational bound.

## Frozen implementation amendment before production

The historical plan did not yet freeze concrete seeds and optimizer budgets.
Before any Phase 40 production result, fix them as follows:

- correlated seeds: `40001`, `40002`, `40003`;
- NOCI seeds: `40101`, `40102`, `40103`;
- correlated initialization: canonical occupied orbitals, seeded `1e-3`
  occupied--virtual noise, and zero correlator amplitudes;
- correlated optimization: 200 Adam steps at learning rate `0.03`, followed by
  at most 80 strong-Wolfe LBFGS iterations at learning rate `0.5`;
- NOCI optimization: 200 variable-projection Adam steps from the existing
  canonical-plus-seeded-random initialization at learning rate `0.01` down to
  `1e-5`, followed by 80 strong-Wolfe LBFGS iterations at learning rate `0.5`;
- optimization quadrature `Q=96`; uniform final audits at `Q=128` and `Q=160`;
- every completed point writes a deterministic tensor checkpoint and the
  aggregate runner supports resume without rerunning completed points;
- no retries, replacement seeds, extra `D/P/K` values, backup ansatz, or
  threshold changes after production output is observed.

## Validation before and during production

- materialized uncorrelated state/exterior-coefficient equivalence;
- AD versus central finite differences for both carrier and correlator
  parameters, with absolute difference at most `1e-6` for each checked scalar;
- antisymmetry residual at most `1e-12` for every reported state;
- `Q=128` to `Q=160` energy change at most `1e-7` and relative norm change at
  most `1e-8`;
- independent reconstruction of serialized observables;
- explicit reporting of every `Q^2`, `D^N`, full alternating tensor, and
  virtual-path materialization.

## Differentiator rule

The primary comparison is the best preregistered `P=5` correlated result
against the best preregistered `K=4` NOCI result at the same `D`. A `D` point
counts as an advantage only when:

1. the correlated reference error plus its `Q=128`--`Q=160` uncertainty is at
   most one half of the NOCI reference error;
2. the correlated raw optimized-parameter count does not exceed the NOCI raw
   optimized-parameter count;
3. at least two of the three correlated seeds meet the same error inequality;
4. all symmetry, gradient and quadrature checks pass.

Passing requires two consecutive `D` values that count as advantages. Wall
time and peak RSS remain separately reported; a parameter-envelope pass is not
misdescribed as a matched-time pass. Projected Slater-rank growth and an energy
below same-basis CI are supporting observations only.

## Failure and publication rule

If the differentiator is absent, unstable, or dependent on an undisclosed
cost mismatch, reject this route and report the negative result. Phase 40 does
not create a second manuscript whether it passes or fails. Only an independent
reproduction of a pass permits a later publication decision.
