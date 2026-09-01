# Active execution plan: Restored Phase 39 N4,D8 numerical addendum

## Numbering note

The original uncommitted Phase 39 plan named this clean-source `N=4,D=8`
calculation. A later manuscript-scope audit also used the Phase 39 label. This
addendum restores the original numerical obligation without deleting or
rewriting the completed scope audit.

## Objective

Run exactly one preregistered clean canonical-Slater `N=4,D=8,Q=128`
soft-Coulomb schedule through `K=1,2,3,4`, including a forced K2
interruption/resume and one clean repeat. Preserve it only as internal
**numerical evidence** for the NOCI-equivalent restricted implementation.

## Frozen scope

- Configuration: `docs/experiments/configs/phase39_n4_d8_k4.json`.
- Registration decision: ADR 0031.
- Continuous harmonic functional basis, physical-SVD interaction, CPU
  complex128, canonical lowest-orbital source, and the public solver command.
- Dense exterior CI is opened only after both production lineages finish.
- No historical FEMPS checkpoint, D6-padded state, CI vector, best-seed search,
  hidden retry, `D>8`, `N>4`, or additional small-system point.

## Required checks

1. Force the registered lineage to stop after K2 and resume through K4.
2. Run one uninterrupted clean lineage with the identical schedule.
3. Independently reconstruct every serialized exterior state, norm, energy,
   variance, particle-TT ranks, source/operator identity, candidate choice,
   and acceptance decision.
4. Report D8 `K` convergence separately from the D6-to-D8 basis change.
5. Explicitly report zero production virtual-path and `D^N` enumeration,
   structural/materialized antisymmetry residual, time, and peak RSS.

## Scientific boundary

Passing or failing this calculation does not establish a new ansatz, an
advantage over NOCI, continuum convergence, automatic stopping, or scaling.
After the artifact and verifier are frozen, no more small NOCI-equivalent
numerical points are authorized. The main line immediately becomes the theory
closure of the single combined manuscript A.
