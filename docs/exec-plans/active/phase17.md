# Active execution plan: Phase 17 unbounded interaction and scaling gate

## Objective

Remove the finite sine-box bottleneck identified by Gate D and determine
whether the continuous ordered-distance functional TN remains accurate and
tractable beyond `N=4`. Build an interaction representation compatible with an
unbounded half-line basis, control any MPO compression globally, and measure
accuracy-to-basis and accuracy-to-bond requirements before making a scaling
claim.

## Checkpoints

- [ ] Derive and test a soft-Coulomb separation suitable for unbounded positive
  gaps, prioritizing Laplace/Gaussian sums whose local odd-Hermite matrix
  elements have independent quadrature checks.
- [ ] Cross-check the unbounded interaction operator against the finite sine
  interval and direct one-/two-distance quadrature on small systems.
- [ ] Compare sine, odd-Hermite, and any generalized-Laguerre candidate at the
  same N=2 and N=4 truth points, with independent `D` and scale controls.
- [ ] Reduce the raw `O(N^2 K)` interaction MPO cost only if a small-system
  global operator/action error is measured; do not certify compression from
  local discarded singular values.
- [ ] Separate optimization from representation capacity using independent
  TT-SVD/DMRG diagnostics and blind global-AD runs at matched `D,chi`.
- [ ] Benchmark at least one controlled N=6 interacting point without a
  product-basis state gather, recording wall time, peak memory, MPO bond, MPS
  bond, norm, gradients, and all approximation controls.
- [ ] Revisit the Li--Waintal comparison after the unbounded and N=6 results;
  retain the narrow naming and priority boundary from ADR 0006.
- [ ] Issue Gate E on unbounded-basis accuracy and larger-particle scaling.

## Exit criterion

Gate E passes only if an unbounded interacting basis improves or matches the
finite sine-box accuracy with independently controlled separation and
quadrature errors, any MPO compression has a global audit, and a blind N=6
native run has a complete basis/scale/bond/optimization budget. Failure keeps
Gate D valid at controlled small-system scope but stops larger-N solver claims.
