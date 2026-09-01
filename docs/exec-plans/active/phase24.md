# Active execution plan: Phase 24 controlled approximate exterior gate

## Objective

Determine whether approximation can evade the exact permanent obstructions
without breaking antisymmetry or replacing a quantitative guarantee by a
heuristic. No variational implementation is admitted until the target error,
complexity, and conditioning assumptions are explicit.

## Candidate K1

Approximate squared norm and functional one-/factorized-two-body observables
directly from compact APG/matrix-pair data while preserving the exterior state
definition exactly. Randomized and deterministic routes are both admissible,
but must expose failure probability, variance/conditioning, and observable
error propagation.

## Checkpoints

- [ ] Define additive/relative state, squared-norm, and energy-error targets,
  including behavior near zero norm and the Rayleigh-quotient denominator.
- [ ] Audit approximation complexity for nonnegative permanents, signed/complex
  permanents, APG overlaps/RDMs, selected pairing schemes, and approximate
  tensor-network contraction using primary sources.
- [ ] Determine which hard reductions survive the proposed physical input
  restrictions and which admit randomized approximation.
- [ ] Derive an a posteriori bound that propagates norm and Hamiltonian
  contraction errors to a certified energy interval.
- [ ] Construct exact small instances spanning positive, cancelling, ill-
  conditioned, and physically optimized regimes; heuristics must be checked
  against full exterior truth.
- [ ] Require exact antisymmetry by construction and quantify estimator bias,
  variance, failure probability, time, and memory jointly in all controls.
- [ ] Audit novelty against APG pair-selection/low-rank methods, Monte Carlo
  geminal evaluation, stochastic TN contraction, and sign-problem literature.
- [ ] Issue Gate K before GPU/AD solver work. Border rank or favorable typical
  samples alone cannot pass.

## Exit criterion

Gate K passes only with a polynomial-cost algorithm under explicit assumptions
and a non-asymptotic state/observable error certificate strong enough to bound
the variational energy. A proof that the needed signed/complex regime retains a
complexity or variance obstruction is an acceptable negative outcome.
