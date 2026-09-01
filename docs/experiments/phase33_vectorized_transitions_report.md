# Phase 33 vectorized transition and backend report

## Scope

Phase 33 integrates the independently verified Phase 32 `N=6` convergence
evidence into the restricted-method manuscript and replaces Python loops over
the `K^2` determinant-transition pairs and physical operator-SVD factors by
batched tensor operations. The pairwise implementation remains available as
the reference route, and exactly singular transition pairs retain the
singular-safe minor fallback.

This is one matched `N=6,D=10,K=4,L=19` backend audit. It is numerical evidence,
not an asymptotic scaling or GPU-acceleration claim.

## Registered results

- CPU batched/reference Hamiltonian maximum difference: `8.899e-16`.
- CPU batched/reference orbital-gradient maximum difference: `4.687e-13`.
- Blackwell/CPU Hamiltonian maximum difference: `1.429e-14`.
- Blackwell/CPU orbital-gradient maximum difference: `3.682e-12`.
- Median CPU reference forward/backward kernel: `0.2803773 s`.
- Median CPU batched forward/backward kernel: `0.0080277 s`, a `34.926x`
  reference speedup.
- Median Blackwell batched forward/backward kernel: `0.0181411 s`, a `15.455x`
  reference speedup.
- The matched 160-Adam/80-L-BFGS solves produce identical dense energy
  `25.050223374041963`. CPU takes `5.0845 s`; Blackwell takes `11.9605 s` and
  peaks at `20,747,264` device bytes.
- Structural antisymmetry residual, norm, conditioning, and all registered
  value/gradient/backend gates pass on both devices.

Blackwell is therefore an admitted production backend, but CPU remains the
default for this workload because the full Blackwell solve is `2.352x` slower.
Seed 3304 misses the separate Phase 29 direct-CI quality control by reaching
error `5.835e-4`; it is retained only as the preregistered matched backend
workload. Phase 32 remains the physics-convergence evidence.

## Reproduction

Primary artifact:
`docs/experiments/results/phase33_vectorized_transitions.json`.

Independent verifier:
`scripts/verify_phase33_vectorized_transitions.py`.
