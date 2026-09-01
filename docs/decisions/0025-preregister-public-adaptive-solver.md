# ADR 0025: Preregister the public bounded adaptive-solver closure

- Status: accepted before production
- Date: 2026-09-01
- Depends on: ADR 0024 and accepted Phase 35

## Context

Phase 35 validates three manually orchestrated truth-free K4-to-K6 lineages,
but benchmark scripts still own the select/optimize/checkpoint sequencing. A
practical minimum solver requires that workflow to be a public library
operation with explicit identities, seeds, resource records, and stage-level
resume. Phase 35 observes no stop event, so an automatic stopping rule remains
unadmitted.

## Decision

Add a public bounded adaptive diagonal-path FEMPS operation. The caller must
provide:

- a source `(K,D,N)` first-quantized continuous functional-basis state;
- one-body and factorized two-body operators with an explicit identity;
- a finite external `max_terms > K`;
- one candidate-pool seed and one optimizer seed for every consecutive target
  K through that maximum;
- a checkpoint path.

The API must reject incomplete schedules, changed source/operator identities,
unbounded or non-increasing maxima, hidden retries, and non-CPU orchestration
in this phase. It writes an atomic outer checkpoint after each completed K and
retains the existing versioned per-K optimizer checkpoints.

## Preregistered physical reproduction

Use the accepted Phase 32 `N=6,D=12,K=4` source and the same Q128 physical-SVD
soft-Coulomb operator as Phase 35. Run exactly one public API schedule:

- target K5: candidate seed 3511, optimizer seed 3511;
- target K6: candidate seed 3512, optimizer seed 3512;
- candidate pool size 32;
- 160 Adam plus 80 L-BFGS steps per target;
- overlap threshold `1e-10`, condition cap `1e8`;
- mandatory external `max_terms=6`.

Interrupt the public call after K5, load and validate the outer checkpoint,
then resume it for K6. The Phase 35 artifact must not be opened until the
public K6 result is complete. No rescue start, seed change, optimizer change,
N/D expansion, or automatic stop is allowed.

## Acceptance gates

- The public result and outer checkpoint validate under their versioned
  contracts and finish at the external K6 cap.
- The partial call stops at K5; the resumed call completes exactly one new
  stage and preserves the K5 stage record.
- K5/K6 selected candidate indices equal frozen Phase 35 lineage 1.
- K5/K6 energies agree with frozen Phase 35 lineage 1 within `1e-11`.
- Every stage reports zero structural antisymmetry residual, zero enumerated
  virtual paths, zero materialized `D^N` coefficients, norm error below
  `1e-10`, condition below `1e8`, time below 600 seconds, and sampled peak RSS
  below 2 GiB.
- Independent reconstruction from committed source/final orbital records
  reproduces factorized energies, exterior norms, variances, candidate
  selections, and source/operator hashes.
- Small-system materialization, exterior energy, AD gradient, invalid-schedule,
  changed-identity, and interruption/resume tests pass before this physical
  run is admitted.

## Scientific boundary

Passing this ADR establishes a reusable bounded orchestration API for one
restricted FEMPS subclass. It does not validate automatic stopping, generic
matrix-wedge contraction, asymptotic scaling, runtime superiority, N8, or a
new state class beyond nonorthogonal multideterminant expansions.
