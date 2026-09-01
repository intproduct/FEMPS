# Phase 35 adaptive-pool stability and stopping calibration

## Decision

Phase 35 **passes the preregistered adaptive-pool stability gate** at the fixed
interacting `N=6,D=12` benchmark. Three fresh, independently seeded candidate
pools all continue monotonically from the accepted `K=4` state through `K=5`
and `K=6`. The automatic stopping rule is **not admitted**, because every one
of the six growth decisions says `continue`; agreement without an observed
stop does not validate a stopping event. Subsequent adaptive calculations must
therefore retain an explicit external maximum `K`.

This is bounded numerical evidence for the restricted nonbranching
diagonal-path FEMPS subclass. It is not an asymptotic, generic-contraction,
runtime-superiority, or `N=8` result.

## Frozen protocol

- Common source: Phase 32 `N=6,D=12,K=4` accepted state.
- Fresh seed pairs: `(3511,3512)`, `(3521,3522)`, `(3531,3532)`.
- Candidate pool: 32 at each `K=4 -> 5` and `K=5 -> 6` step.
- Optimization: 160 Adam plus 80 L-BFGS iterations on CPU.
- Selection inputs: factorized determinant-transition Hamiltonian, overlap,
  generalized-eigenvalue energy, and balanced overlap conditioning.
- Dense CI and the Phase 34 final errors were unavailable until all six
  optimizations were frozen.

## Results

| Seed pair | `K=5` energy | `K=6` energy | `K=6` CI error | `K=6` variance | condition |
|---|---:|---:|---:|---:|---:|
| 3511 / 3512 | 25.049431562122 | 25.049399173589 | 3.27575e-5 | 3.89800e-4 | 2.750 |
| 3521 / 3522 | 25.049434999852 | 25.049400347160 | 3.39311e-5 | 4.15119e-4 | 2.787 |
| 3531 / 3532 | 25.049435352210 | 25.049404050601 | 3.76345e-5 | 4.54299e-4 | 3.072 |

The three `K=6` energies span `4.87701e-6`. Their minimum total improvement
over the common `K=4` energy is `6.70940e-5`. All norm errors are below
`1.34e-15`; structural antisymmetry residuals, virtual-path enumerations, and
materialized `D^N` particle tensors are exactly zero in every admitted record.
Each optimization takes 5.26--6.77 CPU seconds, with maximum sampled RSS
`677,519,360` bytes.

Every predicted fixed-span improvement and every fully reoptimized gain is at
least the preregistered `1e-8` continuation threshold. Thus all six predicted
and realized decisions agree as `continue`. Since no predicted or realized
decision is `stop`, this experiment calibrates continuation consistency but
does not establish a reliable automatic termination rule.

## Independent verification

`scripts/verify_phase35_adaptive_pool_stability.py` reconstructs the physical
quadrature basis, same-basis 924-dimensional CI comparator, common source, and
all six optimized exterior states from committed coefficient records. It
recomputes energies, variances, norms, ordinary particle-TT ranks, storage,
candidate selections, source nesting, stopping decisions, and every acceptance
gate. The verifier reports `verified: true`.

Authoritative artifact:
`docs/experiments/results/phase35_adaptive_pool_stability.json`.

## Scientific boundary and next action

The result replaces the earlier single-pool caveat by three-new-pool stability
evidence, but it does not justify an unbounded adaptive search. The next solver
milestone should package the now-verified growth schedule into a public,
checkpointed adaptive API with deterministic seed sequencing and a mandatory
external `max_K`, rather than expand `N`, `D`, or high-dimensional form-rank
searches.
