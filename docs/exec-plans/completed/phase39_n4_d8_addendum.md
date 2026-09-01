# Completed execution plan: Restored Phase 39 N4,D8 numerical addendum

## Outcome

The single ADR-0031 `N=4,D=8,Q=128,K1--K4` schedule was run once with a
forced K2 interruption/resume and once cleanly. The independent verifier
reconstructs both lineages exactly. The aggregate gate **failed** and the
failure is retained without a rescue run.

## Frozen result

- Energies at `K=1,2,3,4`:
  `11.025186887292186`, `11.023574685785562`,
  `11.023376907553928`, `11.023315453809996`.
- Selected candidates: `0`, `2`, `29`.
- Clean/resume maximum energy difference: `0`.
- D8 same-basis CI energy: `11.023278984749750`.
- Final CI error: `3.6469060e-5` (registered maximum `1e-6`).
- Final variance: `3.2251715e-4` (registered maximum `1e-5`).
- Every structural and materialized antisymmetry residual: `0`.
- Optimizer failures and production virtual-path/`D^N` enumeration: `0`.
- Resumed total / clean times: `18.15 s` / `16.70 s`.
- Maximum sampled process RSS: `657,555,456` bytes.

All gates except `final_accuracy_pass` pass. Within that combined gate, both
the final CI error and variance fail. The K sequence is monotone and stable,
but fixed `K=4` with the frozen budget is not accurate enough at D8.

## Scientific boundary

This is internal **numerical evidence** for a finite NOCI-equivalent
calculation. It is not a new ansatz, a continuum result, or an advantage over
NOCI. The historical D6 comparison used a different registered schedule, so
its `-5.2226e-4` final-energy change is descriptive rather than a pure
same-optimizer D-convergence claim.

No additional small numerical point, rescue seed, or enlarged budget is
authorized. The next active work is manuscript-A theory closure.
