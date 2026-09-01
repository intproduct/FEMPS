# ADR 0019: Admit same-point N=6 multiseed stability gate

- Status: accepted
- Date: 2026-09-01
- Depends on: ADR 0018

## Context

The single-seed `N=6,D=10,K=4` pilot passes with direct CI error `1.382e-4`,
variance `1.197e-3`, 85.59 s runtime, and 798 MB sampled peak RSS. The next
uncertainty is optimization stability, not basis order, particle count, or
correlation capacity. A same-point multiseed gate costs materially less and
preserves stronger truth than changing several axes at once.

## Decision

Run exactly three blind K4 starts with seeds 31, 37, and 43. Every run uses the
same `N=6,D=10,Q=128` Hamiltonian, physical-operator SVD, 160 Adam steps, 80
L-BFGS steps, learning-rate schedule, and generalized-eigenproblem threshold.
No CI state, K1 checkpoint, or seed-specific hyperparameter may initialize or
tune these runs.

The committed ADR-0018 direct quadrature CI record supplies the fixed reference;
the physical operator and factorized finite-basis reference are recomputed.
Seed 31 performs the bounded million-coefficient materialization and ordinary
particle-TT audit. Seeds 37 and 43 skip that exponential validator but must
report structural antisymmetry residuals.

## Preregistered acceptance

Every seed must satisfy:

- same-basis CI error in `[-1e-9,5e-4]`;
- energy variance at most `5e-3`;
- norm error at most `1e-10`;
- structural antisymmetry residual at most `1e-12`;
- all four generalized-overlap directions retained and retained condition
  number at most `1e8`;
- zero virtual-path and production particle-tensor enumeration;
- factorized/direct reference disagreement and operator reconstruction error at
  most `1e-11`;
- sampled peak process RSS at most 1.5 GiB and total wall time at most 600 s.

Seed 31 additionally requires materialized antisymmetry residual at most
`1e-12`. Across all runs, energy spread must not exceed `2.5e-4`.

## Consequences

- Passing admits reproducible N6 feasibility at this one `D,K` point only.
- Failure leaves N6 as single-seed feasibility evidence; no extra rescue seed
  or post-result seed-specific tuning is allowed.
- Even a pass does not authorize N8. The next decision must compare physics
  value and measured `N,K,L` costs against existing N4 results before changing
  size again.

## Validation update (2026-09-01)

All three blind runs pass. CI errors are `2.494e-4`, `2.932e-4`, and
`4.031e-4`; maximum variance is `3.094e-3`; energy spread is `1.537e-4`.
Every run retains rank four with maximum condition number 4.424, zero norm and
structural antisymmetry residuals, and zero path enumeration. Seed 31 has zero
million-coefficient materialized residual. Maximum time is 112.14 s and peak
RSS 882,962,432 bytes. N expansion stops pending the matched N4-to-N6 cost and
structure audit required above.
