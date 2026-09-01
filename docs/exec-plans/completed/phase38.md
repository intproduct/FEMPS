# Completed execution plan: Phase 38 Clean-Source Seed Robustness

## Objective

Determine whether the Phase 37 clean canonical-Slater solver result is stable
under two independently preregistered candidate-pool and optimizer schedules at
the same N4,D6 soft-Coulomb point.

## Closure evidence

- [x] ADR 0027 and two complete machine-readable schedules were pushed before
  either fresh production result was opened.
- [x] Schedule A was interrupted at K2, resumed to K4, and repeated cleanly;
  selected candidates `1/13/22` and every stored energy agree exactly.
- [x] Schedule B completed cleanly with distinct candidates `31/26/25`.
- [x] Including Phase 37, final K4 energies have spread `2.035e-9`; maximum
  fresh same-basis CI error is `2.523e-9` and maximum variance `1.462e-8`.
- [x] The optimizer failure count and outcome-dependent retry count are zero.
- [x] Independent exterior reconstruction verifies all 12 complete-run states,
  candidate selections, energies, variances, norms, ordinary particle-TT ranks,
  source/operator identities, and aggregate gates.
- [x] Every structural and materialized antisymmetry residual is zero, with no
  production virtual-path or `D^N` coefficient enumeration.
- [x] All command times are below 9 seconds and sampled peak RSS is below 643
  MB on the registered machine.

## Decision

Phase 38 is **PASS as bounded clean-source schedule robustness at one N4,D6
point**. It does not admit universal seed independence, automatic stopping,
N/D scaling, runtime superiority, or generic matrix-wedge contraction.

Authoritative artifact:
`docs/experiments/results/phase38_clean_source_seed_robustness.json`.

Independent verifier:
`scripts/verify_phase38_clean_source_seed_robustness.py`.
