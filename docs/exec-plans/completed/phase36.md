# Completed execution plan: Phase 36 Public Adaptive Solver Closure

## Objective

Convert the verified Phase 34--35 staged growth workflow into one public,
checkpointed FEMPS solver operation that runs a bounded truth-free adaptive K
schedule without manual script orchestration.

## Closure evidence

- [x] Added versioned adaptive result and outer-checkpoint schemas.
- [x] Added explicit target-K, candidate-seed, and optimizer-seed stage records.
- [x] Required a finite external `max_terms > source K` and a complete
  consecutive schedule; automatic stopping is `not_admitted`.
- [x] Bound resume to source-orbital and operator hashes plus the complete
  adaptive/optimizer configuration.
- [x] Wrote an atomic outer checkpoint after every completed K while retaining
  existing per-K optimizer checkpoints.
- [x] Added invalid-schedule, changed-identity, materialized exterior energy,
  antisymmetry, AD gradient, and interruption/resume tests.
- [x] Preregistered ADR 0025 and froze the physical runner before production.
- [x] Interrupted the N6,D12 lineage after K5 and resumed K6 through the public
  API. Selected candidates `12/3` and stored energies reproduce frozen Phase 35
  lineage 1 exactly.
- [x] Independent reconstruction gives final CI error `3.27575e-5`, variance
  `3.89800e-4`, norm error `1.11e-16`, and zero structural antisymmetry residual.
- [x] Production enumerates zero virtual paths and zero `D^N` coefficients;
  K5/K6 stay below 7 seconds and 664 MB sampled RSS.

## Failure record

The retained first wrapper attempt performed one redundant QR before entering
the public API. Its K6 energy differed from the frozen result by `3.60423e-11`
and failed the preregistered `1e-11` match gate. The gate was not relaxed.
Passing production removes only the redundant wrapper QR; seeds, Hamiltonian,
optimizer, external K cap, and tolerances are unchanged.

## Decision

Phase 36 is **PASS as a reusable bounded orchestration API** for the restricted
nonbranching first-quantized continuous functional-basis FEMPS. It does not
admit automatic stopping, a generic matrix-wedge solver, runtime superiority,
asymptotic scaling, or N8.

Authoritative artifact:
`docs/experiments/results/phase36_public_adaptive_solver.json`.

Independent verifier:
`scripts/verify_phase36_public_adaptive_solver.py`.
