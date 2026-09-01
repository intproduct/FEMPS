# Completed execution plan: Phase 33 N6 Evidence Integration and Transition Vectorization

## Objective

Integrate the independently verified Phase 32 N6 D/K convergence into the
restricted-method manuscript and remove the measured small-kernel backend
bottleneck without changing the first-quantized continuous FEMPS definition or
weakening exact contraction and antisymmetry guarantees.

## Completion record (2026-09-01)

- [x] Added the Phase 32 D/K convergence, resource, variance, antisymmetry,
  CI/Slater/ordinary-TT, and limitation evidence to the manuscript and
  reproduction manifest.
- [x] Vectorized well-conditioned transition pairs and physical factor axes;
  retained the historical pairwise reference and exact singular-safe fallback.
- [x] Added value and reverse-mode gradient parity tests for both regular and
  exactly singular transitions.
- [x] CPU batched/reference differences are `8.899e-16` in the Hamiltonian and
  `4.687e-13` in gradients.
- [x] Blackwell/CPU differences are `1.429e-14` and `3.682e-12`.
- [x] Median forward/backward time decreases from `0.2803773 s` to
  `0.0080277 s` on CPU (`34.926x`).
- [x] Matched CPU and Blackwell solves give identical dense energy
  `25.050223374041963`; both pass resource, norm, antisymmetry, and conditioning
  gates.
- [x] Blackwell is admitted, but CPU stays default because the full solve takes
  `5.0845 s` on CPU and `11.9605 s` on Blackwell.
- [x] The manuscript PDF is rebuilt as a clean nine-page artifact, and all
  numerical claims are linked to the ten-entry reproduction manifest.
- [x] The final repository suite passes: `261 passed`, with one pre-existing
  latticeTN scalar-reporting warning and no test failures.

## Scientific decision

Phase 33 is **PASS as a restricted-solver implementation and backend result**.
It establishes correctness, portability, and a large reference-kernel speedup,
not generic FEMPS scalability or GPU acceleration. The preregistered backend
seed misses the older direct-CI quality gate and is not physics evidence; the
independent Phase 32 convergence artifact retains that role.

Primary record:
`docs/experiments/results/phase33_vectorized_transitions.json`.

Independent verifier:
`scripts/verify_phase33_vectorized_transitions.py`.
