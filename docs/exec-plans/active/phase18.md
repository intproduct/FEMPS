# Active execution plan: Phase 18 basis efficiency and N=8 gate

## Objective

Reduce the N=6 half-line basis error exposed by Gate E and determine whether
the unbounded ordered-distance functional TN admits a controlled N=8 point
without temporary dense raw-MPO storage becoming the dominant resource.

## Checkpoints

- [ ] Compare boundary-compatible half-line bases or scale-adaptive mixtures
  against odd Hermite at matched N=2, N=4, and N=6 orders; admit a new basis
  only after analytic overlap/kinetic checks and independent interaction
  quadrature.
- [ ] Extend N=6 Galerkin basis/scale controls beyond D=8 or produce a bounded
  alternative truth audit that resolves the observed `1.33e-2` basis error.
- [ ] Avoid or reduce dense `W^2 D^2` raw Fourier MPO storage while preserving
  the exact four-state recurrence; audit any new structured/compressed builder
  globally on bounded systems.
- [ ] Improve global-AD convergence using only predeclared schedules and
  separate the effect from MPS capacity by TT-SVD or an independent local
  optimizer.
- [ ] Run a controlled N=8 interacting point with complete `D`, scale,
  Fourier/local-quadrature, MPO-bond, MPS-bond, optimization, wall-time, and
  peak-memory budgets and no product-basis training gather.
- [ ] Reassess the accuracy-to-basis and accuracy-to-resource trend from
  N=2,4,6,8 before making any scaling statement.
- [ ] Issue Gate F on basis efficiency and N=8 admission.

## Exit criterion

Gate F passes only if the dominant N=6 basis error is measurably reduced with
independent controls, the production MPO path no longer relies on an
unaudited dense raw representation, and at least one blind N=8 point meets a
predeclared total-error and optimization budget. Failure retains Gate E only
at controlled N<=6 scope.
