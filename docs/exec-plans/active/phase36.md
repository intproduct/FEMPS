# Active execution plan: Phase 36 Public Adaptive Solver Closure

## Objective

Convert the verified Phase 34--35 staged growth workflow into one public,
checkpointed FEMPS solver operation that can run a bounded truth-free adaptive
correlation schedule without manual script orchestration. Preserve exact
antisymmetry, first quantization, continuous functional bases, and the
restricted nonbranching diagonal-path carrier.

## Frozen scope

- Keep the physics point at interacting `N=6,D=12`; do not expand N or D.
- Expose a mandatory external `max_K`; do not claim automatic stopping.
- Use deterministic per-stage candidate and optimizer seeds supplied by the
  caller. No hidden retries or post-result seed changes.
- Reuse the admitted factorized transition backend and existing optimizer.
- Do not restart high-dimensional form-rank searches.

## Primary work

1. Define a public adaptive-run result/checkpoint schema with source identity,
   current K, seed schedule, selected candidate indices, predicted gains,
   optimizer records, resource records, and scientific-boundary fields.
2. Implement a bounded `K_start -> max_K` driver around the existing
   truth-free candidate selector and diagonal-path optimizer.
3. Support stage-level checkpoint/resume, including an interruption test that
   is bitwise or tolerance-equivalent to an uninterrupted run.
4. Report norm error, antisymmetry residual, condition, time, memory, and zero
   virtual-path/particle-tensor enumeration at every stage.
5. Add a small materialized exterior equivalence and AD gradient test before
   any performance tuning.
6. Reproduce one preregistered Phase 35 lineage end to end through the public
   API, then compare it with the frozen artifact using an independent verifier.

## Acceptance gates

- Public API requires `max_K > K_start` and an explicit complete seed schedule;
  invalid or incomplete schedules fail before optimization.
- The API reproduces the selected indices, energies, and diagnostics of one
  Phase 35 lineage within registered float tolerances.
- Interrupted/resumed and uninterrupted runs agree within `1e-11` in energy
  and preserve identical stage choices.
- Small-system materialization, factorized energy, and automatic-differentiation
  gradients pass existing exact tolerances.
- Every output stage explicitly reports antisymmetry residual and resource
  counters; no production stage enumerates virtual paths or a `D^N` tensor.
- A committed artifact, independent verifier, standard tests, and method
  evidence lint pass.

## Failure rule

If a reusable API cannot reproduce the frozen lineage without script-specific
state or hidden manual intervention, report the workflow as a benchmark-only
prototype and do not describe it as a minimum usable adaptive solver. Failure
does not authorize N/D expansion, unbounded K search, or pure-mathematics work.
