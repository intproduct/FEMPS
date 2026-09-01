# Active execution plan: Phase 28 FEMPS Algorithm and Physics Recovery

## Status and objective

**Status:** algorithm-recovery gate passed on 2026-09-01; the paper-third-draft
track remains active. This plan supersedes Phase 27 for project priority
without weakening any established no-go theorem or changing the first-
quantized continuous functional-basis definition of FEMPS.

Deliver a minimal FEMPS solver that is runnable, independently verifiable, and
able to optimize at least one nontrivial interacting continuous fermion model.
Mathematical classification work stays parked unless it directly blocks an
algorithmic or physical acceptance criterion below.

## Route boundary

The primary route is the exactly contractible **nonbranching diagonal-path
FEMPS**. A global virtual label `a=1,...,K` is conserved at every site, giving

\[
  \Psi=\sum_{a=1}^{K} c_a\,
  u_{a1}\wedge\cdots\wedge u_{aN}.
\]

This is an exact matrix-wedge FEMPS embedding with structured correlation bond
`chi=K`. It contains a single Slater at `K=1`, supports nonorthogonal
multideterminant states for increasing `K`, and contracts through `K^2`
determinant/Slater--Condon transitions rather than `K^(N-1)` virtual paths.
The scientific tradeoff must be stated plainly: this route is a restricted
FEMPS/selected nonorthogonal-CI family, not a solution of generic FEMPS exact
contraction.

The sole backup route is a controlled stochastic estimator for generic
matrix-wedge cores. It remains inactive until it has an explicit estimator,
failure probability, observable/energy error or variance bound, and a measured
antisymmetry residual. Finite AGP/Pfaffian and ordered-sector solvers remain
named controls/backends; neither is relabeled as the primary FEMPS method.

## A. Algorithm feasibility audit

- [x] Inventory the current state definition, exact materialization oracle,
  functional bases, Hamiltonians, AD optimizers, checkpoint support, and prior
  benchmark records.
- [x] Record the generic exact squared-norm obstruction and the boundary it
  leaves for restricted structure or controlled approximation.
- [x] Select one primary route and one inactive backup in ADR 0017.
- [x] Implement robust overlap, one-body, and factorized two-body transition
  matrices for diagonal-path FEMPS. A condition-number-gated determinant/
  solve path automatically falls back to singular-safe minors, and both paths
  pass independent value, reverse-mode, and finite-difference checks.
- [x] Measure time and peak memory against the declared `(N,D,K,L)` cost model,
  where `L` is the two-body operator factorization rank.
- [x] Decide after E1--E4 that the route is a useful exact restricted baseline
  with systematic correlation control, but not yet a demonstrated scalable or
  novel alternative to nonorthogonal selected CI.

## B. Minimal usable solver

- [x] Continuous functional basis and one-/two-body operator assembly.
- [x] Deterministic `K=1` Slater and known `K>1` multideterminant embedding.
- [x] Norm, energy, and energy-variance calculation or controlled estimate.
- [x] Reverse-mode gradients checked against full exterior materialization;
  finite-difference auditing remains part of the performance gate.
- [x] Variational optimization with deterministic seed, checkpoint/resume,
  best-state restoration, and stable overlap conditioning.
- [x] Every record includes structural and, when feasible, materialized
  antisymmetry residuals; approximate calculations additionally include error,
  variance, and failure-probability fields.
- [x] Independent `D` and `K=chi` convergence driver for E1/E2; no coupled sweep may
  substitute for the two one-axis checks.
- [x] Production contraction never enumerates all virtual paths or materializes
  the full antisymmetric coefficient tensor. Materialization remains a bounded
  truth oracle only.

## C. Ordered physics gates

1. [x] `N=2` noninteracting spinless harmonic fermions.
2. [x] `N=2` interacting harmonic or soft-Coulomb model with analytic or
   independently converged reference.
3. [x] `N=4` noninteracting system, including the `K=1` FEMPS versus ordinary
   particle-TT exchange-rank comparison.
4. [x] `N=4` interacting continuous fermions. Three blind `D=6,K=4` runs and
   three truth-free nested-basis `D=7,K=4` continuations pass the registered
   error/variance/symmetry criteria; CPU RSS and CI/Slater/AGP/ordinary-TT
   comparators are complete.
5. [ ] Larger `N` or more realistic interactions only after gates 1--3 pass.

Each point reports energy and reference error, variance/uncertainty, norm
error, antisymmetry residual, `D` and `K` convergence, wall time, peak memory,
optimization stability, and comparisons with exact diagonalization,
Slater/CI, and ordinary particle TT. Second-quantized DMRG may be an explicitly
external comparator.

## D. Paper-third-draft track

- [ ] Preserve exact-rank, Slater flat-spectrum, and fixed-small-bond
  complexity results as algorithm-design constraints rather than a denial of
  every approximate or structured FEMPS method.
- [ ] Distinguish exact-contraction hardness from restricted and stochastic
  possibilities.
- [ ] Address both reviewer rounds on R1/R2 provenance, FCI/DMRG comparisons,
  AGP embedding, odd/even forms, numerical wording, and AI disclosure.
- [ ] Remove Phase/Gate/internal evidence labels from the submission version.
- [ ] Do not require new high-dimensional four-form results for draft three.

## Complexity contract for the primary route

For `K` determinants with orbital matrices of shape `D x N`:

- stored parameters: `O(K D N)` using diagonal structure;
- all overlaps: `O(K^2 (D N^2 + N^3))` time and `O(K^2 + K D N)` memory;
- dense one-body preprocessing: `O(K D^2 N)`, followed by `K^2` transitions;
- factorized two-body operator of rank `L`: polynomial in
  `K^2 L (D^2 N + N^3)` on the well-conditioned path;
- singular-safe determinant-minor references may add factors of `N` or `N^2`
  and are validation paths, not hidden exponential algorithms.

Every implementation report must give its actual formula rather than citing
this upper-level summary alone.

## Success and stop criteria

Phase 28 succeeds only when at least one nontrivial interacting model is
stably optimized, energy converges systematically in `D` or `K`, an independent
script reproduces the result, all approximation errors/variances and
antisymmetry residuals are explicit, and no forbidden enumeration occurs. It
must also demonstrate either a practical FEMPS advantage or a clear measured
tradeoff.

If these conditions fail, report that the tested FEMPS form is not a general
practical solver and pivot to the registered restricted subclass or an
explicitly renamed first-quantized alternative. Additional pure mathematics
does not postpone this decision.

## Immediate milestones

1. Land the route ADR, feasibility audit, and project-priority corrections.
2. Add exact diagonal-path transition contractions plus full-materialization
   and gradient tests.
3. Reproduce E1 and E2 in one stable result schema with checkpoints.
4. Pass E3, then run the first controlled E4 `(D,K)` grid.

All four immediate milestones are complete. The accepted E4 result and its
independent verifier are documented in
`docs/experiments/phase28_e4_closure_report.md`. The next numerical gate is a
nonquadratic soft-Coulomb transferability test; it must not weaken the current
truth, variance, symmetry, memory, or comparator requirements.
