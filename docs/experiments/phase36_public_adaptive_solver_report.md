# Phase 36 public bounded adaptive-solver closure

## Decision

Phase 36 **passes** as a public, checkpointed orchestration closure for the
restricted nonbranching diagonal-path FEMPS solver. The library now accepts an
explicit finite K schedule, performs truth-free candidate selection and
single-K optimization, records every stage, and resumes from an outer
stage-level checkpoint. It never selects its own terminal K: a caller-supplied
external `max_terms` remains mandatory and automatic stopping is explicitly
`not_admitted`.

This is bounded numerical and software evidence for a first-quantized
continuous functional-basis FEMPS subclass. It is not a generic contraction,
asymptotic scaling, runtime-superiority, or N8 result.

## Public contract

The new public API exports:

- `AdaptiveDiagonalPathStageConfig` for target K, candidate seed, and optimizer
  seed;
- `AdaptiveDiagonalPathConfig` for a complete consecutive schedule and finite
  external `max_terms`;
- `run_bounded_adaptive_diagonal_path` for select/optimize/checkpoint stages;
- versioned adaptive checkpoint/result constants, loaders, and validators.

Each outer checkpoint binds the source orbital hash, operator hash and label,
adaptive configuration, optimizer template, current K, canonical current
orbitals, and all completed stage records. Every stage retains its existing
single-K optimizer checkpoint and reports energy, norm, antisymmetry,
conditioning, time, memory, and structural operation counters.

## Small-system gates

The pre-production suite verifies:

- incomplete or non-increasing K schedules are rejected;
- a changed source identity is rejected on resume;
- interrupted K2-to-K3-to-K4 execution agrees with uninterrupted execution
  within `1e-11` and selects identical candidates;
- factorized and materialized exterior energies agree below `1e-10`;
- factorized and explicit exterior AD gradients agree below `1e-8`;
- every bounded materialization has antisymmetry residual below `1e-12`;
- every production stage reports zero virtual paths and zero `D^N`
  coefficient materialization.

## Preregistered N6,D12 reproduction

ADR 0025 freezes the Phase 35 lineage-1 seeds and optimizer budgets:

| Target | Candidate seed | Optimizer seed | Selected candidate | Energy |
|---|---:|---:|---:|---:|
| K5 | 3511 | 3511 | 12 | 25.049431562122003 |
| K6 | 3512 | 3512 | 3 | 25.049399173588654 |

The public call is deliberately interrupted after K5, its outer checkpoint is
loaded and validated, and the resumed call completes exactly the K6 stage. The
Phase 35 artifact is opened only after K6 is frozen. Both selected indices and
both stored energies then match frozen Phase 35 lineage 1 exactly in the JSON
float representation.

Independent reconstruction from committed source, K5, and K6 orbital records
gives:

- K5 energy `25.049431562122017`, CI error `6.51460e-5`, variance
  `7.50965e-4`, norm error `4.44e-16`;
- K6 energy `25.049399173588696`, CI error `3.27575e-5`, variance
  `3.89800e-4`, norm error `1.11e-16`.

K5/K6 take 6.54/6.28 CPU seconds with sampled peak RSS 656,551,936 and
663,908,352 bytes. Structural antisymmetry residual, enumerated virtual paths,
and materialized `D^N` production coefficients are zero at both stages.

## Failed engineering attempt retained

The first wrapper attempt is retained as
`phase36_public_adaptive_solver_attempt1.json`. It canonicalized an already
canonical source once before entering an API that canonicalizes its input,
adding an unnecessary QR gauge projection. K5 and both candidate choices still
matched, but the K6 energy differed from frozen Phase 35 by `3.60423e-11`, so
it failed the preregistered `1e-11` equality gate. The gate was not relaxed.
The wrapper was changed to pass the raw accepted checkpoint orbitals so the
public API performs exactly one source canonicalization; seeds, model,
optimizer, K cap, and all tolerances remained frozen.

## Evidence

- Accepted artifact:
  `docs/experiments/results/phase36_public_adaptive_solver.json`.
- Retained failed attempt:
  `docs/experiments/results/phase36_public_adaptive_solver_attempt1.json`.
- Independent verifier:
  `scripts/verify_phase36_public_adaptive_solver.py`.

## Next action

The public API is no longer benchmark-script-specific, but the admitted N6 run
still starts from a preoptimized K4 checkpoint. The next practical milestone
should expose an end-to-end command that initializes a canonical Slater source,
optimizes it, and executes a bounded adaptive schedule on an interacting model
without requiring a historical benchmark checkpoint.
