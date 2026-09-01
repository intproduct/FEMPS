# Completed execution plan: Phase 37 End-to-End Slater-Source Solver Command

## Objective

Deliver one user-executable command that constructs a canonical single-Slater
source for an interacting continuous model, optimizes it, and runs the public
bounded adaptive FEMPS schedule to a caller-supplied finite maximum K without a
historical FEMPS checkpoint.

## Closure evidence

- [x] Added versioned clean-source result and checkpoint schemas.
- [x] Required explicit model/configuration records, deterministic source,
  candidate, and optimizer seeds, checkpoint path, result path, and finite
  external `max_K`.
- [x] Constructed the N4,D6,Q128 soft-Coulomb operator and canonical
  lowest-orbital Slater solely from registered inputs.
- [x] Optimized K1 and passed its accepted state into the existing bounded
  adaptive K2--K4 API without CI or historical FEMPS initialization.
- [x] Bound resume to the complete configuration, source identity, operator
  identity, schema, and canonical initial-source identity.
- [x] Added small explicit-exterior materialization, energy, antisymmetry, AD
  gradient, interruption/resume, and changed-configuration tests.
- [x] Forced a K2 interruption and resumed through K4; clean and resumed
  energies agree exactly at every K.
- [x] Independent reconstruction gives final energy
  `11.023837713691632`, same-basis CI error `4.883e-10`, variance
  `2.863e-9`, norm error `3.33e-16`, and zero structural antisymmetry
  residual.
- [x] Production enumerates zero virtual paths and zero `D^N` coefficients;
  clean K1--K4 execution takes 5.95 seconds and peaks at 648,269,824 bytes
  sampled process RSS on the registered machine.
- [x] Added a committed artifact, independent verifier, fourteenth
  reproduction-manifest entry, method evidence mapping, and manuscript claim.

## Decision

Phase 37 is **PASS as one bounded end-to-end interacting FEMPS solver point**.
It closes the historical correlated-source dependency at N4,D6 and preserves
first quantization, the continuous functional basis, exact structural
antisymmetry, and separate D/K semantics. It does not admit automatic stopping,
seed robustness, N/D scaling, runtime superiority, or generic matrix-wedge
contraction.

Authoritative artifact:
`docs/experiments/results/phase37_slater_source_solver.json`.

Independent verifier:
`scripts/verify_phase37_slater_source_solver.py`.
