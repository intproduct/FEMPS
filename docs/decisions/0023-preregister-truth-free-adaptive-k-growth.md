# ADR 0023: Preregister truth-free adaptive K growth at N6,D12

- Status: accepted before production
- Date: 2026-09-01
- Depends on: ADR 0017, ADR 0022, Phase 32 D/K convergence

## Context

Phase 33 makes determinant-transition optimization inexpensive enough that the
next uncertainty is algorithmic rather than backend-related: can correlation
multiplicity be increased using only the current FEMPS state and Hamiltonian,
without CI-informed initialization or a blind proliferation of determinants?

The existing `extend_diagonal_path_terms` preserves nesting but adds seeded
random terms without ranking them. A practical restricted solver needs an
auditable growth rule and a cold-start control.

## Preregistered decision

At the frozen interacting `N=6,D=12` soft-Coulomb point, start from the accepted
Phase 32 K4 checkpoint. For K4-to-K5 and K5-to-K6, draw exactly 32 seeded
orthonormal Slater candidates with seeds 3451 and 3452. Rank candidates only by
the lowest fixed-span determinant-transition generalized eigenvalue. Reject a
candidate if the augmented balanced overlap loses a direction, exceeds
condition `1e8`, or violates variational nesting by more than `1e-10`.

Optimize each selected state for the frozen 160 Adam plus 80 L-BFGS schedule.
Run one same-budget cold K6 control with seed 3460. No CI energy, CI vector,
dense exterior Hamiltonian, or materialized particle tensor may be constructed
until all three optimizations are frozen.

## Gates and stopping rule

- Adaptive K4/K5/K6 energies must be nonincreasing within `1e-9`.
- Norm error is at most `1e-10`, structural antisymmetry residual at most
  `1e-12`, retained condition at most `1e8`, and production path/tensor
  enumeration remains zero.
- Each optimized point stays below 600 seconds and 2 GiB sampled CPU RSS.
- Final dense truth must reproduce factorized energies within `1e-10`; operator
  and quadrature gates remain `1e-11` and `2e-12`.
- A total K4-to-K6 improvement of `1e-8` is labeled measurable. A smaller
  improvement is reported as saturation, not rescued with extra seeds.
- Existing gauge-balanced pruning is assessed at K6 with energy tolerance
  `1e-7`. If it triggers, Phase 34 remains open until the deletion and required
  reoptimization are separately preregistered; it is not silently applied.
- The cold K6 comparison is diagnostic, not a gate manufactured to favor the
  adaptive route.

Failure stops N expansion and records whether the limitation is candidate
selection, nonlinear optimization, conditioning, resource use, or K saturation.

## Scientific boundary

This is numerical evidence for one restricted first-quantized continuous FEMPS
subclass and one physical point. It cannot establish generic FEMPS efficiency,
asymptotic scaling, or superiority over CI, particle TT, or DMRG.
