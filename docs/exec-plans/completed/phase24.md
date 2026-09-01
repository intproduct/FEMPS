# Completed execution plan: Phase 24 controlled approximate exterior gate

## Objective

Determine whether approximation can evade the exact permanent obstructions
without breaking antisymmetry or replacing a quantitative guarantee by a
heuristic.

## Completed checkpoints

- [x] Defined additive/relative state, squared-norm, and energy-error targets,
  including zero norm and the Rayleigh-quotient denominator.
- [x] Audited nonnegative, real/Hermitian PSD, and arbitrary complex permanent
  approximation; APG selection; stochastic TN contraction; and sign-problem
  context using primary sources.
- [x] Separated the JSV entrywise-nonnegative FPRAS from the admitted real-PSD
  hard family.
- [x] Proved that a generic relative squared-norm PRAS would yield a real-PSD
  permanent PRAS by the Phase 22 APG identity.
- [x] Derived an a-posteriori norm/numerator-to-energy bound and simultaneous
  confidence interval.
- [x] Added exact positive, cancelling, precision-ill-conditioned, signed-PSD,
  and positive/negative energy controls.
- [x] Made bias, variance/tail, failure probability, conditioning, time, and
  memory mandatory for any successor estimator.
- [x] Classified existing optimized LC-AGP points as tractable controls rather
  than evidence for generic APG approximation; no post-failure GPU run was
  admitted.
- [x] Issued Gate K and ADR 0014.

## Result

Gate K is **FAIL** for generic relative Candidate K1 unless `RP=NP`. Exact
antisymmetry remains intact, but generic relative norm estimation already
contains real-PSD permanent approximation. Additive estimates cannot uniformly
control the variational denominator under cancellation.

The result is not universal: entrywise-nonnegative inputs, a certified norm
lower bound, or a separate structural promise may still support an approximate
method. Such a method must exclude the hard embedding and provide a
non-asymptotic energy certificate.
