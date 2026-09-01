# Active execution plan: Phase 32 N6 Independent Convergence and Resource Boundary

## Objective

Close the most important remaining numerical limitation of the restricted
nonbranching FEMPS solver: establish independent functional-basis (`D`) and
correlation-multiplicity (`K`) convergence for the interacting `N=6`
soft-Coulomb model without increasing particle number or weakening the
first-quantized continuous functional-basis definition.

## Primary route

Continue the exact polynomial determinant-transition subclass selected by
ADR-0017.  Use direct exterior CI only as a same-basis reference and never as
an initializer.  Production contractions must not enumerate virtual paths or
materialize the `D^N` particle tensor; small materialization remains a
validation-only antisymmetry check.

## Registered experiments

1. Audit quadrature and physical operator-SVD errors at `D=8,10,12` with fixed
   soft-Coulomb parameters and an independently generated direct-CI reference.
2. Register an `N=6` `K` axis at fixed basis, minimally `K=1,2,4`, with identical
   optimization budgets and at least three blind seeds at the decisive point.
3. Register an `N=6` `D` axis at fixed `K=4`, minimally `D=8,10,12`; basis growth
   may transfer the optimized lower-`D` orbitals but must remain truth-free.
4. Measure energy, variance, norm error, both antisymmetry residuals when
   materialization is admitted, conditioning, wall time, and sampled peak RSS.
5. Report direct CI, single Slater, and ordinary particle-TT rank/storage
   comparators at every admitted point.  DMRG remains deferred while direct CI
   is exact and cheaper in the registered truth spaces.

## Resource and claim boundaries

- Do not start `N=8` or a new mathematical rank search in this phase.
- Use a written memory/time estimate before every `D=12` production run.
- Stop a branch if estimated peak RSS exceeds the registered local cap or if
  conditioning repeatedly invalidates the generalized eigensolve.
- Two or three basis points are convergence evidence, not an asymptotic fit.
- Every floating-point conclusion remains **numerical evidence**.
- Any failure to obtain monotone or interpretable convergence is a solver
  limitation to report, not a reason to substitute pure mathematics.

## Acceptance gates

- `K`-axis variational energies are nonincreasing within registered numerical
  tolerance, with variance and optimization stability reported.
- The `D` axis shows an interpretable trend relative to a clearly named
  finite-basis or continuum proxy; no continuum bound is implied without proof.
- Structural antisymmetry residual is at most `1e-12` at every point, and every
  admitted materialized residual is at most `1e-12`.
- No production result enumerates virtual paths or materializes `D^N`
  coefficients.
- At least one independent verifier reconstructs all table values from raw
  artifacts and rejects changed seeds, tolerances, or evidence labels.
- The method manuscript is revised only after these gates pass; otherwise its
  explicit `N=6` basis-convergence limitation remains unchanged.

## Parallel bounded paper task

Maintain the structural/no-go third-draft revision as a bounded secondary task:
address the registered reviewer points without adding high-dimensional
four-form searches, exploratory floating-point claims, Phase/Gate labels, or
internal evidence tags to the submission manuscript.
