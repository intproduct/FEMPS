# ADR 0031: Preregister the restored Phase 39 N4,D8 internal point

## Status

Accepted on 2026-09-02 before any clean-source `N=4,D=8` production result.

## Context

The original Phase 39 plan required one clean-source functional-basis
extension from `D=6` to `D=8`. It was displaced by the manuscript-scope
correction before its seeds and gates were committed. The user now requires
that single point to be completed and then forbids further expansion of the
small NOCI-equivalent numerical grid.

## Frozen configuration

The machine-readable source of truth is
`docs/experiments/configs/phase39_n4_d8_k4.json`:

- `N=4`, `D=8`, `Q=128`, harmonic trap and basis;
- unit-coupling, unit-softening soft Coulomb with physical-SVD threshold
  `1e-13`;
- source seed `4001`;
- K2/K3/K4 candidate/optimizer pairs `4011/4012`, `4021/4022`, and
  `4031/4032`;
- 60 Adam steps plus at most 40 L-BFGS refinement steps at every K;
- pool size 32 and external cap `K=4`.

The optimizer budget is inherited unchanged from the clean D6 command. The
new seeds are consecutive values chosen before viewing a D8 clean-source
result and are not selected from Phase 37/38 outcomes.

## Frozen gates

- resumed and clean lineages complete and have identical candidate choices;
- stage energies agree within `1e-11` and are nonincreasing within `1e-9`;
- K1 CI error/variance are at most `2e-3`/`1e-2`;
- K4 CI error/variance are at most `1e-6`/`1e-5`;
- norm error at most `1e-10`, both antisymmetry residuals at most `1e-12`,
  physical factorization error at most `1e-11`;
- every stage at most 180 s, each lineage at most 900 s, sampled peak RSS at
  most 2 GiB;
- no optimizer failure, production virtual-path enumeration, or production
  `D^N` materialization.

## Execution and failure rule

The registered lineage is forced to stop after K2, then resumes through K4;
one identical clean lineage follows. Dense D8 CI and the historical D6
internal comparator are opened only after both lineages are frozen. Every
outcome is retained. Failure does not authorize changed seeds, larger budgets,
a rescue run, a second D8 schedule, or another small point.

## Publication boundary

This result is internal **numerical evidence**. It may inform which bounded
NOCI-equivalent number is selected for manuscript A, but it cannot support a
method-paper claim. Manuscript B remains closed until a non-NOCI explicit-
correlation `D`-convergence advantage or a matched Li--Waintal/same-basis-DMRG
tradeoff is independently reproduced.
