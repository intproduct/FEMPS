# Active execution plan: Phase 37 End-to-End Slater-Source Solver Command

## Objective

Deliver one user-executable command that constructs a canonical single-Slater
source for an interacting continuous model, optimizes it, and runs the public
bounded adaptive FEMPS schedule to a caller-supplied finite maximum K. Remove
the admitted physical workflow's dependency on a historical preoptimized
checkpoint while preserving the Phase 36 contracts.

## Frozen scope

- Use an existing N4 soft-Coulomb truth region; do not expand N or D.
- Start from the canonical lowest-orbital Slater, not CI data or a saved FEMPS
  source.
- Require explicit model parameters, deterministic source/candidate/optimizer
  seeds, output path, checkpoint path, and external `max_K`.
- Keep CPU default and the exact restricted determinant-transition backend.
- No automatic stopping, hidden retries, post-result schedule changes, N8, or
  high-dimensional form-rank search.

## Primary work

1. Add a stable command/config record that constructs the functional-basis
   Hamiltonian and canonical Slater source from user inputs.
2. Optimize the K1 source through the existing public single-K solver, then
   pass its accepted orbitals into `run_bounded_adaptive_diagonal_path`.
3. Support command-level checkpoint/resume and deterministic regeneration from
   no prior runtime artifact.
4. Report per-stage energy, variance, norm, antisymmetry, D/K, time, memory,
   source/operator identities, and zero path/particle-tensor enumeration.
5. Preregister one bounded N4 run and compare it with same-basis CI, the source
   Slater, ordinary particle TT, and the existing manually orchestrated N4
   reference.

## Acceptance gates

- A clean invocation with no historical FEMPS checkpoint completes K1 through
  the explicit maximum K and writes validated results/checkpoints.
- A resumed invocation preserves stage choices and agrees with a clean run
  within `1e-11` in energy.
- Energy is nonincreasing with K within `1e-9`; final same-basis CI error and
  variance satisfy preregistered bounds derived before production.
- Every stage reports structural antisymmetry residual at most `1e-12`, norm
  error at most `1e-10`, zero virtual paths, and zero production `D^N`
  materialization.
- Small-system command materialization and AD checks precede production.
- A committed artifact, independent verifier, reproduction-manifest entry,
  standard tests, and method evidence lint pass.

## Failure rule

If the clean Slater-source command cannot reproduce a stable interacting K
sequence without hidden benchmark state, report the public API as requiring an
expert-provided correlated source and do not call it an end-to-end solver.
Failure does not authorize threshold relaxation, seed replacement, N/D
expansion, or pure-mathematics work.
