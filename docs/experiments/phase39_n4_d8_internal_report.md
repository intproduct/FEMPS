# Restored Phase 39 N4,D8 internal numerical report

## Decision

The preregistered calculation is independently reproducible but fails its
frozen final-accuracy gate. It is retained as **internal numerical evidence**
and closes further small NOCI-equivalent numerical expansion.

## Frozen calculation

The model is four spinless fermions in a unit harmonic trap with unit-coupling,
unit-softening one-dimensional soft Coulomb interaction. The calculation uses
the first eight harmonic functions, `Q=128` physical-SVD interaction,
complex128 CPU arithmetic, a canonical lowest-orbital Slater source, and
adaptive nonbranching determinant paths through the externally fixed cap
`K=4`.

The schedule was committed before production in ADR 0031. The resumed lineage
was forced to stop after K2; a clean lineage used the identical source,
candidate, and optimizer seeds. Dense D8 CI and the historical D6 control were
opened only after both D8 lineages completed. There were no outcome-dependent
retries.

| K | Energy | Error vs D8 CI | Variance | particle-TT ranks |
|---:|---:|---:|---:|---|
| 1 | 11.025186887292186 | 1.9079025e-3 | 9.8733938e-3 | 4, 6, 4 |
| 2 | 11.023574685785562 | 2.9570104e-4 | 1.9735150e-3 | 8, 12, 8 |
| 3 | 11.023376907553928 | 9.7922804e-5 | 7.8197114e-4 | 8, 18, 8 |
| 4 | 11.023315453809996 | 3.6469060e-5 | 3.2251715e-4 | 8, 24, 8 |

The independently reconstructed D8 CI energy is
`11.023278984749750`. Clean and resumed energies agree exactly at every K and
select candidates `0/2/29`.

## Gate audit

All registered gates pass except `final_accuracy_pass`. The final CI error
exceeds `1e-6`, and the variance exceeds `1e-5`. Norm errors are at most
`4.45e-16`; all structural and materialized antisymmetry residuals are zero;
optimizer failures and production virtual-path/`D^N` enumeration are zero.
The physical factorization error is `1.2634e-15`. Resumed total and clean
times are `18.15 s` and `16.70 s`; maximum sampled RSS is `657,555,456` bytes.

The D8 final energy is `5.2226e-4` below the historical D6 Phase 37 final
energy. Because the D6 and D8 schedules use different registered seeds, this
is descriptive basis-change evidence, not a pure matched-optimizer D-axis or
a continuum claim.

## Interpretation

The finite `K=4` NOCI-equivalent representation improves systematically but
does not meet the preregistered D8 accuracy/variance target. No rescue seed,
larger K, longer budget, or additional small point is admitted. The result is
not evidence for a distinct FEMPS method.
