# Active execution plan: Phase 29 Restricted FEMPS N6 Decision

## Objective

Determine whether the frozen diagonal-path FEMPS solver supports reproducible
larger-particle interacting physics at `N=6` without weakening exact
antisymmetry, first quantization, the continuous functional basis, independent
truth, or resource reporting.

This is a bounded algorithm decision, not an invitation to reopen generic
contraction theory or high-dimensional form-rank searches.

## Fixed representation and contract

- State: nonbranching diagonal-path FEMPS, exactly a `K`-term nonorthogonal
  Slater sum inside matrix-wedge FEMPS.
- Basis: continuous harmonic-oscillator functional basis.
- Hamiltonian: one-body harmonic trap plus soft-Coulomb pair interaction.
- Public API and schemas: contract version 1 in
  `docs/DIAGONAL_PATH_SOLVER_CONTRACT.md`.
- Production: exact `K^2` determinant transitions with the physical-operator
  SVD; no virtual-path or full particle-tensor enumeration.
- Validation: direct dense-quadrature exterior CI and bounded materialization
  only where declared feasible.

## Completed entry gate

- [x] Freeze result schema v2 and checkpoint schema v1 with executable
  validators and public checkpoint loading.
- [x] Record ADR 0018 with pre-run memory, time, accuracy, variance, symmetry,
  factorization, and correlation-improvement limits.
- [x] Run the single-seed `N=6,D=10,K=1 -> 4` pilot.
- [x] Independently verify every acceptance decision.

The pilot passes: K4 same-basis CI error is `1.382e-4`, 5.31% of the K1 error;
variance is `1.197e-3`; both antisymmetry residuals are zero; runtime is 85.59 s
and sampled peak RSS is 798,416,896 bytes. The full evidence is in
`docs/experiments/phase29_n6_soft_coulomb_pilot_report.md`.

## Active gate: same-point reproducibility decision

Before any larger `N`, `D`, or `K`, decide whether a three-seed stability gate
at exactly `N=6,D=10,K=4` is justified by the measured cost and physics value.
The decision must be recorded in an ADR before additional optimization runs.

If admitted, the gate must:

1. retain the direct exterior-CI energy and physical-operator factorization;
2. use three genuinely blind deterministic seeds, not truth-derived states;
3. report every energy, variance, norm error, structural antisymmetry residual,
   time, peak RSS, retained determinant rank, and conditioning diagnostic;
4. materialize at least one final state for the million-coefficient
   antisymmetry and ordinary particle-TT audit;
5. report energy spread and worst-case CI error/variance;
6. stop if any run exceeds the ADR resource cap or becomes rank/conditioning
   unstable;
7. make no N8, asymptotic, runtime-advantage, or generic FEMPS claim.

## Stop and pivot rule

If multiseed N6 stability fails, keep N4 as the reproducibly validated scope
and treat N6 as a single-seed feasibility point. Do not add seeds or tune each
seed separately after seeing the result. The allowed next study would be a
clearly named restricted-subclass optimization/stabilization task, not pure
mathematical classification.

## Stability-gate result and current milestone

ADR 0019 admits and fixes seeds 31, 37, and 43 before execution. All three
blind K4 points pass with maximum same-basis CI error `4.031e-4`, maximum
variance `3.094e-3`, energy spread `1.537e-4`, maximum condition number 4.424,
maximum runtime 112.14 s, and maximum sampled RSS 882,962,432 bytes. Structural
antisymmetry residuals are zero for all points; the registered seed-31
million-coefficient residual is also zero.

The active milestone is now a matched N4-to-N6 structural and contraction-cost
audit at fixed `D=10,K=4,L=19`. It must separate operation-count scaling from
different optimizer budgets and report that direct CI remains faster in the
current truth region. No N8 diagonal-path calculation is authorized.
