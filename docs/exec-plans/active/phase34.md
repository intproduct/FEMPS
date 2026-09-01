# Active execution plan: Phase 34 Adaptive N6 Correlation Growth

## Objective

Turn the Phase 33 contraction speedup into a solver capability: implement and
audit truth-free adaptive growth and optional pruning of the diagonal-path
correlation multiplicity `K` at the interacting `N=6,D=12` point. Determine
whether `K=4 -> 5 -> 6` gives reproducible energy/variance improvement without
CI initialization, virtual-path enumeration, or loss of exact antisymmetry.

## Primary work

1. Specify a deterministic K-growth/pruning contract using only current-state
   energy, gradients, overlaps, and conditioning; CI data may be read only in
   the final audit.
2. Add small-system materialization and AD-gradient checks for every new growth
   or pruning operation before performance work.
3. Run a preregistered `N=6,D=12` K4-to-K5-to-K6 continuation with fixed seeds,
   optimizer schedule, time/memory caps, and checkpointed raw exterior
   coefficients.
4. Compare adaptive continuation with a same-budget cold K6 start and the
   existing Slater, direct-CI, and ordinary particle-TT controls.
5. Independently rebuild energies, variances, norms, conditioning, storage, and
   TT ranks from the saved exterior coefficients.

## Boundaries

- No N8 expansion, high-dimensional form-rank search, or generic exact-FEMPS
  contraction claim.
- No CI vector, energy, or coefficient may influence term selection,
  initialization, pruning, stopping, or hyperparameters.
- CPU is the default; Blackwell may be used only under ADR 0022 matched parity
  and resource reporting.
- If K5/K6 gives no stable improvement, record saturation and stop. Do not add
  rescue seeds, expand N, or silently change optimizer budgets.
- The method remains first-quantized, continuous functional-basis FEMPS.

## Acceptance gates

- Growth/pruning value and reverse-mode gradients match explicit exterior truth
  on bounded small systems to `1e-10` and `1e-8`, respectively.
- Every admitted N6 point has norm error at most `1e-10`, structural
  antisymmetry residual at most `1e-12`, retained condition number at most
  `1e8`, and zero production virtual-path/particle-tensor enumeration.
- K4-to-K5-to-K6 energies are nonincreasing under the frozen continuation; any
  pruning step may increase energy only within a preregistered tolerance and
  must be followed by reoptimization.
- Runtime, sampled RSS, device memory when applicable, variance, CI error, and
  independent D/K interpretation are reported for every point.
- A cold-start control distinguishes adaptive-growth benefit from merely using
  a larger K.
- A source-hashed artifact and independent verifier reproduce every accepted
  numerical claim; full tests and manuscript evidence lint remain passing.
