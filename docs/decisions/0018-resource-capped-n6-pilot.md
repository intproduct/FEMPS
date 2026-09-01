# ADR 0018: Admit one resource-capped N=6 soft-Coulomb pilot

- Status: accepted
- Date: 2026-09-01
- Depends on: ADR 0017 and diagonal-path solver contract version 1

## Context

Phase 28 passes its restricted interacting criteria through `N=4,D=12,K=5`.
Before increasing particle count, the solver API, result schema, checkpoint
identity, antisymmetry reporting, and reproduction rules are frozen. The next
question is not generic scaling; it is whether the same exact restricted
algorithm remains numerically controlled at one larger particle count while
independent truth is still available.

`N=6,D=10` is the largest useful bounded pilot before the validation geometry
changes qualitatively:

- exterior dimension `binom(10,6)=210`, so direct dense-quadrature CI remains
  independently feasible;
- the complete particle tensor has `10^6` complex coefficients. Its raw
  complex128 storage is 16,000,000 bytes and it can be materialized once for
  validation, but never in production contractions;
- the accepted physical soft-Coulomb operator backend at `D=10,Q=128` has
  rank 19 and dense reconstruction error `1.34e-15`;
- at `K=4`, production stores 240 orbital scalars, evaluates 16 transition
  pairs and 9,120 factorized two-body determinant minors, and enumerates zero
  virtual paths.

This keeps truth stronger than a large-`N` energy-only run. It also gives a
particle-TT comparator through the bounded materialization oracle.

## Decision

Proceed with exactly one registered `N=6,D=10` pilot containing:

1. a blind `K=1` Slater optimization;
2. exact blind term embedding of that checkpoint into `K=4` by retaining the
   first determinant and adding three fixed-seed random Slaters;
3. a direct unfactorized `Q=128` dense exterior-CI reference;
4. a physical-operator-SVD production backend and an independent dense
   reconstruction audit;
5. bounded validation-only `10^6`-coefficient materializations of the final
   `K=4` state and CI reference for antisymmetry and particle-TT ranks;
6. CPU execution in complex128, deterministic seeds and checkpoints.

No additional seed may be added to rescue a failed pilot. A passing pilot is
feasibility evidence only; multiseed stability would require a later separate
gate.

## Preregistered resource and acceptance limits

- sampled peak process RSS at most 1.5 GiB per point;
- total wall time at most 600 s per point;
- factorization and factorized/direct finite-basis reference disagreement at
  most `1e-11`;
- norm error at most `1e-10`;
- structural and materialized antisymmetry residuals at most `1e-12`;
- no virtual-path enumeration or production particle-tensor materialization;
- `K=4` energy no greater than the nested initial energy or `K=1` source by
  more than `1e-9`;
- `K=4` same-basis CI error at most `5e-4` and at most half the `K=1` error;
- `K=4` energy variance at most `5e-3`.

The pilot passes only if an independent verifier recomputes all decisions from
the committed artifact. Exceeding a resource cap or missing a materialized
residual is a failed pilot, not an invitation to weaken the threshold.

## Consequences

- **Go** is limited to this one truth-controlled point. It does not admit
  `N>6`, `D>10`, asymptotic scaling, or a practical advantage claim.
- If the pilot passes, compare its measured `K^2 L` cost and particle-TT ranks
  with the N4 audit before designing any multiseed or basis-axis gate.
- If it fails, retain N4 as the validated scope and record whether the cause is
  optimization, correlation capacity, materialization, or resources.
- High-dimensional alternating-form work remains parked because it does not
  decide this pilot.

## Validation update (2026-09-01)

The registered pilot passes. Direct `Q=128` CI gives energy
`25.049639839832263`. Blind `K=1` has error `2.603e-3`; exact blind growth to
`K=4` reduces it to `1.382e-4`, or 5.31% of the source error, with variance
`1.197e-3`. Both antisymmetry residuals are zero. The K4 point takes 85.59 s
and peaks at 798,416,896 bytes, below both resource caps. Its ordinary
particle-TT ranks `(10,45,80,45,10)` remain below CI only at the center cut,
whose rank is 120. This admits a same-point multiseed go/no-go decision, not a
larger N or scaling claim.
