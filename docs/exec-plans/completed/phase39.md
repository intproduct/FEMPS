# Completed execution plan: Phase 39 Combined-Manuscript Closure and Distinctiveness Audit

## Outcome

Phase 39 closed the two-paper drift. The project has one publication
manuscript, `math/femps_no_go_manuscript.tex`. The former restricted-method
draft is a frozen internal reproduction note, not Paper B and not a submission
candidate.

The combined manuscript now contains:

- Structural results I--III, preserving the original Theorems 1--3 as three
  directly visible statements;
- the exact bond boundary: pointwise hardness at `chi=2`, the proved signed
  exact-norm reduction at maximum bond three, and the `chi=2` exact-norm
  sharpening only as a conjecture;
- the nonbranching diagonal-path calculations as an explicitly
  NOCI-equivalent numerical exercise, without an independent method claim.

## Completed audit

- [x] Rebuilt and visually checked the 14-page combined review PDF.
- [x] Passed proof/source and single-manuscript scope tests.
- [x] Froze all current diagonal-path claims at the NOCI-equivalent boundary.
- [x] Audited explicit correlation, Li--Waintal, same-basis CI/DMRG, tensor
  backflow, and VMC prior art.
- [x] Selected one candidate only: a symmetric explicit correlator multiplying
  an exterior carrier (ADR 0029).
- [x] Implemented a bounded `N=2` materialization and AD prototype with an
  independent artifact verifier.

## Exploratory prototype result

The `N=2` soft-Coulomb pilot is **numerical evidence**, not a preregistered
benchmark. It verifies exact antisymmetry on the audited grid, an AD/finite-
difference gradient difference of about `1.51e-11`, and increasing projected
Slater rank through `D=12`. Its energy is `2.55383312944`, about `4.48e-6`
above the independent relative-coordinate reference. These facts select a
candidate experiment; they do not establish a functional-basis convergence
advantage, scalability, novelty, or a new paper.

## Publication rule carried forward

Algorithm experiments continue without drafting a second manuscript. A new
method manuscript may be considered only after a frozen, independently
reproduced experiment demonstrates either:

1. explicit-correlation `D`-convergence beyond optimized fixed-`K` NOCI; or
2. a genuine measured tradeoff against Li--Waintal and/or same-basis DMRG.

Failure leaves all numerical results inside the combined structural/no-go
paper as bounded algorithm-design evidence.
