# Phase 38 Clean-Source Seed Robustness Report

**Evidence label:** numerical evidence.

## Decision

Phase 38 is **PASS** for two additional preregistered clean-source schedules at
the fixed N4,D6,Q128 soft-Coulomb point.  The fresh lineages choose different
candidate paths yet converge to the same small energy neighborhood.  This
establishes bounded schedule robustness at one physical point; it does not
establish universal seed independence, automatic stopping, N/D scaling, or
generic FEMPS contraction.

## Frozen schedules

| Lineage | Source seed | K2 candidate/optimizer | K3 | K4 | Execution |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 3801 | 3811/3812 | 3821/3822 | 3831/3832 | forced K2 interruption, resume, and clean repeat |
| B | 3901 | 3911/3912 | 3921/3922 | 3931/3932 | clean |

ADR 0027 and the two committed JSON configurations were pushed before any
fresh result was opened.  There were zero outcome-dependent retries.

## Reconstructed physics

| Lineage | Selected candidates K2/K3/K4 | K1 energy | K2 energy | K3 energy | K4 energy | K4 CI error | K4 variance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A resumed | 1/13/22 | 11.025279022118207 | 11.023864545278125 | 11.023837714428870 | 11.023837713901010 | 6.977e-10 | 3.877e-9 |
| A clean | 1/13/22 | 11.025279022118207 | 11.023864545278125 | 11.023837714428870 | 11.023837713901010 | 6.977e-10 | 3.877e-9 |
| B clean | 31/26/25 | 11.025279022118207 | 11.023864509150076 | 11.023837718183610 | 11.023837715726264 | 2.523e-9 | 1.462e-8 |

The independently rebuilt A clean/resume energy difference is exactly zero at
every K.  Including the Phase 37 K4 control, the final-energy spread is
`2.035e-9`, against the preregistered `2e-6` bound.  The maximum fresh K4 CI
error is `2.523e-9`, maximum variance is `1.462e-8`, and the optimizer failure
count is zero.

## Structural and resource audit

- Every stored structural and materialized antisymmetry residual is zero; the
  independent exterior reconstruction also gives zero residual for all 12
  complete-run states.
- The maximum independently rebuilt norm error is `6.66e-16`.
- Every production stage reports zero enumerated virtual paths and zero
  materialized labelled-particle `D^N` coefficients.
- The physical-SVD operator factorization error is `1.275e-15`.
- A partial-plus-resumed takes 8.64 seconds, A clean 7.18 seconds, and B clean
  7.31 seconds.  The maximum command peak process RSS is 642,023,424 bytes;
  the maximum optimizer-stage time is 2.86 seconds.
- The ordinary particle-TT ranks remain `(4,6,4)` at K1, `(6,10,6)` at K2,
  and `(6,15,6)` at K3/K4, while FEMPS correlation multiplicity is the explicit
  independent axis K=1,2,3,4.

## Reproduction

```powershell
python scripts/benchmark_phase38_clean_source_seed_robustness.py
python scripts/verify_phase38_clean_source_seed_robustness.py
```

The authoritative artifact is
`docs/experiments/results/phase38_clean_source_seed_robustness.json`.
