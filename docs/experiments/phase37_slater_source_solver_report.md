# Phase 37 Clean Slater-Source Solver Report

Evidence level: **numerical evidence**

## Outcome

Phase 37 passes ADR 0026. One public command now constructs the N4,D6
continuous harmonic-oscillator functional basis, builds the Q128 physical-SVD
soft-Coulomb operator, initializes the canonical lowest-orbital Slater,
optimizes K1, and grows the restricted nonbranching FEMPS through an explicit
external `max_K=4`. It uses no historical FEMPS checkpoint and no CI
initializer.

The registered run was interrupted after K2 and resumed to K4. A separate
clean uninterrupted run used the identical configuration. Their candidate
indices and all four serialized energies agree exactly.

## Registered command

```powershell
python scripts/run_femps_slater_source_solver.py `
  --config docs/experiments/configs/phase37_n4_d6_k4.json `
  --max-k 4 `
  --checkpoint checkpoints/phase37_slater_source_solver/resumed.pt `
  --output docs/experiments/results/phase37_slater_source_solver.json
```

The production benchmark additionally forces the K2 interruption, resumes it,
and performs the clean control. Configuration, initial source, optimized
source, and operator identities are versioned and hash checked at the command
checkpoint boundary.

## Physical results

| K | Energy | Error versus same-basis CI | Variance | Ordinary particle-TT ranks |
|---:|---:|---:|---:|---|
| 1 | 11.025279022118207 | 1.441309e-3 | 6.359820e-3 | (4, 6, 4) |
| 2 | 11.023864516114418 | 2.680291e-5 | 1.929373e-4 | (6, 10, 6) |
| 3 | 11.023837719660035 | 6.456689e-9 | 3.784195e-8 | (6, 15, 6) |
| 4 | 11.023837713691630 | 4.882850e-10 | 2.862870e-9 | (6, 15, 6) |

The same-basis dense-quadrature CI energy is `11.0238377132033`. Energy is
nonincreasing at every registered K. The final command energy is
`6.256214e-8` below the existing manually orchestrated Phase 28 K4 control;
this is an accuracy/reproducibility comparison, not a runtime-superiority
claim.

Every stage has structural antisymmetry residual zero, materialized validation
residual below `1e-12`, norm error at machine precision, zero enumerated
virtual paths, and zero production `D^N` coefficient materialization. The
factorized operator has rank 11 and relative dense reconstruction error
`1.274652e-15`.

## Resume and resources

- selected candidates K2/K3/K4: `23/30/25`;
- maximum clean/resume energy difference: `0`;
- K1+K2 partial-call time: `3.8812 s`;
- K3+K4 resume-call time: `3.3048 s`;
- clean K1--K4 command time: `5.9512 s`;
- sampled command peak RSS: `648,269,824` bytes maximum.

All preregistered time, memory, norm, variance, accuracy, identity, and
antisymmetry gates pass. The independent verifier rebuilds the Q128 dense CI
Hamiltonian, all exterior states, energies, variances, norms, particle-TT
ranks, and seeded candidate selections from the committed artifact.

## Scientific boundary

This result establishes an end-to-end user command for one exactly
contractible, nonbranching, multideterminant FEMPS subclass in a small
interacting continuum truth region. It does not establish automatic stopping,
generic matrix-wedge contraction, asymptotic scalability, determinant-state
novelty, or superiority over CI/DMRG. The external maximum K remains
mandatory.
