# Completed execution plan: Phase 35 Adaptive-Pool Stability and Stop Calibration

## Objective

Determine whether the Phase 34 truth-free K-growth result is stable across
independent seeded candidate pools at the same interacting `N=6,D=12` point,
and assess whether fixed-span predicted improvements provide a defensible
automatic stopping signal. Particle number and basis order remained fixed.

## Frozen workload

- [x] Start every lineage from the same accepted Phase 32 D12,K4 checkpoint.
- [x] Use 32 candidates per growth step and seed pairs `(3511,3512)`,
  `(3521,3522)`, and `(3531,3532)`.
- [x] Use 160 Adam plus 80 L-BFGS iterations on CPU.
- [x] Freeze all six optimizations before constructing dense CI or reading the
  Phase 34 final errors.
- [x] Keep the Phase 34 cold K6 record historical; use no rescue starts.

## Closure evidence

- [x] All three K4-to-K5-to-K6 axes are monotone. Minimum total improvement is
  `6.70940e-5`.
- [x] K6 energies are `25.049399173589`, `25.049400347160`, and
  `25.049404050601`, with spread `4.87701e-6`.
- [x] Maximum K6 same-basis CI error is `3.76345e-5`; maximum variance is
  `4.54299e-4`; maximum balanced condition is `3.072`.
- [x] Norm errors are below `1.34e-15`; structural antisymmetry residuals,
  virtual-path enumeration, and materialized `D^N` tensors are zero.
- [x] Each optimization takes 5.26--6.77 CPU seconds; maximum sampled RSS is
  `677,519,360` bytes.
- [x] The independent verifier reconstructs every exterior state, candidate
  selection, source nesting, stopping decision, and acceptance gate.
- [x] All six predicted and realized decisions agree as `continue`.

## Decision

Phase 35 is **PASS for fixed-point adaptive-pool stability**. The automatic
stopping rule is **not admitted**: no stop event occurred, so agreement among
six continuation decisions cannot validate termination. Adaptive production
must retain a mandatory external `max_K`.

This evidence is bounded to the restricted nonbranching first-quantized
continuous functional-basis FEMPS at `N=6,D=12`. It makes no generic,
asymptotic, runtime-superiority, or N8 claim.

Authoritative artifact:
`docs/experiments/results/phase35_adaptive_pool_stability.json`.

Independent verifier:
`scripts/verify_phase35_adaptive_pool_stability.py`.
