# ADR 0024: Preregister adaptive-pool stability and stopping calibration

- Status: accepted before production
- Date: 2026-09-01
- Depends on: ADR 0023 and accepted Phase 34

## Context

Phase 34 shows a measurable truth-free K4-to-K6 improvement for one seeded
candidate-pool lineage and one cold K6 control. That is algorithmic feasibility,
but it does not establish that fixed-pool selection is stable across seeds or
that its predicted fixed-span improvement can safely stop K growth.

## Preregistered workload

Run three fresh lineages from the same accepted Phase 32 `N=6,D=12,K=4`
checkpoint. Each growth step ranks exactly 32 candidates. The K5/K6 seed pairs
are `(3511,3512)`, `(3521,3522)`, and `(3531,3532)`. Every selected state uses
the unchanged 160 Adam plus 80 L-BFGS CPU schedule. Complete all six
optimizations before constructing dense CI or reading the Phase 34 final
errors. No rescue pool, seed-specific tuning, N/D expansion, or replacement of
a failed lineage is permitted.

## Stability gates

- Every K4/K5/K6 energy axis is nonincreasing within `1e-9` and improves from
  K4 to K6 by at least `1e-8`.
- Every K6 same-basis CI error is at most `1.1e-4`, variance at most `1.5e-3`,
  and the three K6 energies have spread at most `1e-4`.
- Norm error is at most `1e-10`, structural antisymmetry residual at most
  `1e-12`, balanced condition at most `1e8`, and every K direction survives.
- Production path and particle-tensor enumeration are zero. Each point stays
  below 600 seconds and 2 GiB sampled CPU RSS.
- Factorized/dense energy, physical-operator SVD, and Q128/Q160 quadrature gates
  remain `1e-10`, `1e-11`, and `2e-12`.

## Stopping-signal calibration

Preregister the candidate rule “continue if the selected fixed-span predicted
improvement is at least `1e-8`; otherwise stop.” Compare it with the fully
reoptimized energy gain using the same `1e-8` threshold for all six steps.
Record every agreement or disagreement. The rule is admitted as an automatic
stop only if all six decisions agree and at least one registered step actually
produces a stop decision. If all steps correctly say continue, the audit may
pass as a no-false-stop calibration, but K remains externally capped because no
stop event has been validated.

Failure of the stability gates is adaptive-pool instability. Failure or absence
of a validated stop event does not invalidate exact contraction, but it forbids
an automatic stopping or efficiency claim.

## Scientific boundary

This is bounded numerical evidence at one N, D, Hamiltonian, optimizer budget,
and candidate-pool size. It cannot establish asymptotic scaling, generic FEMPS
efficiency, or superiority over direct CI, particle TT, or DMRG.
