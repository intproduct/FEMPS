# Completed execution plan: Phase 32 N6 Independent Convergence and Resource Boundary

## Objective

Close the principal numerical limitation of the restricted nonbranching FEMPS
solver by establishing independent functional-basis (`D`) and correlation-
multiplicity (`K`) convergence for the interacting `N=6` soft-Coulomb model.

## Registered route and boundaries

- Exact determinant-transition diagonal-path FEMPS selected by ADR 0017.
- Direct exterior CI is a same-basis reference, never an initializer.
- Production contractions enumerate no virtual paths and materialize no `D^N`
  particle tensor.
- Registered axes are `K=1,2,4` at `D=10` and `D=8,10,12` at `K=4`.
- Every optimization uses 160 Adam steps plus 80 LBFGS steps.
- The decisive `D=10,K=4` stability evidence remains the three blind Phase 29
  seeds 31, 37, and 43.
- No `N=8` run, asymptotic fit, or high-dimensional form-rank search is admitted.

## Completion record (2026-09-01)

- [x] Physical operator-SVD ranks are 15, 19, and 23 at `D=8,10,12`; relative
  reconstruction errors are below `4e-15`.
- [x] The Q128-to-Q160 dense-operator changes are `2.79e-13`, `6.35e-13`, and
  `1.35e-12`, respectively.
- [x] A written D12 preflight predicted 244.4 s and 1.92 GB CPU RSS against
  registered 600 s and 2 GiB caps.
- [x] A complex128 Blackwell AD probe passed, but the non-vectorized
  `N=6,D=10,K=4` GPU branch was stopped after the registered 600 s limit.
  ADR 0021 therefore fixes CPU as the production backend until the transition
  loops are vectorized.
- [x] At `D=10`, energies decrease monotonically from `25.052242782725`
  (`K=1`) to `25.050276338618` (`K=2`) and `25.049825287522` (`K=4`).
- [x] At `K=4`, energies decrease monotonically from `25.051264850894`
  (`D=8`) to `25.049825287522` (`D=10`) and `25.049471144618` (`D=12`).
- [x] The D12 error against direct same-basis CI is `1.04729e-4`, with energy
  variance `1.12233e-3`.
- [x] Every structural antisymmetry residual is zero. The admitted D8 and D10
  materialized checks are also zero; D12 never materializes `12^6` coefficients.
- [x] The largest production point takes 107.7 s and 974,974,976 sampled RSS
  bytes, below both registered caps.
- [x] Compact exterior-sector matrices reproduce ordinary particle-TT ranks
  without constructing `D^N`. At D12,K4 the ranks are `(12,60,80,60,12)`:
  132,768 TT scalars versus 292 stored diagonal-path FEMPS scalars.
- [x] An independent verifier rebuilds all three Hamiltonians from the raw
  exterior coefficients and reproduces energies, variances, norms, TT ranks,
  storage, source hashes, seeds, thresholds, and acceptance gates.

## Scientific decision

Phase 32 is **PASS as numerical convergence evidence for the restricted
subclass**. It is not an asymptotic scaling, generic FEMPS, or superiority
result. Direct CI remains cheaper and exact in the admitted truth spaces. The
measured advantage is the clean separation between exact antisymmetry and a
small correlation multiplicity, with the explicit tradeoff that the current
transition implementation needs CPU execution or future vectorization.

Primary record:
`docs/experiments/results/phase32_n6_convergence.json`.

Independent verifier:
`scripts/verify_phase32_n6_convergence.py`.
