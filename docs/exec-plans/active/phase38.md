# Active execution plan: Phase 38 Clean-Source Seed Robustness

## Objective

Determine whether the Phase 37 clean canonical-Slater solver result is stable
under independently preregistered candidate-pool and optimizer seed schedules
at the same N4,D6 soft-Coulomb point before any N or D expansion.

## Frozen scope

- Keep N=4, D=6, Q=128, the physical-SVD operator, canonical lowest-orbital
  source, K1--K4 schedule, optimizer budgets, tolerances, and CPU default from
  Phase 37.
- Add exactly two fresh complete candidate/optimizer seed schedules in a new
  preregistration ADR before either production run.
- Use the public clean-source command and its existing checkpoint/result
  schemas; do not add benchmark-only initialization or manual stage surgery.
- Keep dense CI strictly post-optimization and validation-only.
- No seed replacement, threshold relaxation, retries selected by final energy,
  automatic stopping, N/D expansion, generic VMC branch, or form-rank search.

## Primary work

1. Preregister the two fresh schedules and acceptance thresholds derived from
   the already frozen Phase 37 reference, without viewing their outcomes.
2. Run both clean K1--K4 lineages and independently reconstruct every accepted
   exterior state.
3. Report all lineages, including failures, with energy, variance, norm,
   antisymmetry residual, D, K, condition diagnostics, time, peak RSS, selected
   candidates, and enumeration counters.
4. Compare energy spread, final CI errors, convergence shape, and optimizer
   failure incidence with Phase 37; do not select a best seed.
5. Add a committed artifact, independent verifier, report, reproduction entry,
   and method-claim boundary if and only if the registered gates pass.

## Acceptance gates

- Both fresh lineages complete K1--K4 without hidden retry or schedule change.
- Energy is nonincreasing with K within `1e-9` for every lineage.
- Every stage has structural antisymmetry residual at most `1e-12`, norm error
  at most `1e-10`, zero virtual-path enumeration, and zero production `D^N`
  materialization.
- Clean rerun or forced-resume reproduction agrees within `1e-11` in energy
  for at least one fresh lineage.
- Final energy spread, maximum same-basis CI error, maximum variance, and all
  optimization failures are reported against preregistered bounds.
- The independent verifier recomputes acceptance from serialized exterior
  states and model data rather than trusting stored summary flags.

## Failure rule

If either preregistered lineage fails, preserve and report it. Do not replace
the seed or expand the search. Classify the Phase 37 success as schedule
sensitive, keep the public command bounded, and make optimizer/initialization
stability the next gate. Failure does not authorize N/D expansion, automatic
stopping, generic-FEMPS claims, or pure-mathematics work.
