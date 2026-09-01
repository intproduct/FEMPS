# ADR 0022: Admit the vectorized Blackwell backend and retain CPU as default

- Status: accepted
- Date: 2026-09-01
- Supersedes the temporary backend restriction in ADR 0021
- Depends on: ADR 0017, ADR 0021, Phase 33 matched backend gate

## Context

ADR 0021 restricted production to CPU because the Python-loop Blackwell
`N=6,D=10,K=4` workload exceeded 600 seconds. Phase 33 batches all
well-conditioned `K^2` transition pairs and physical factor axes while
retaining a pairwise reference and singular-safe fallback.

The registered CPU batched implementation agrees with the pairwise reference
to `8.899e-16` in the Hamiltonian and `4.687e-13` in reverse-mode orbital
gradients. Blackwell agrees with CPU to `1.429e-14` and `3.682e-12`,
respectively. Both matched 160-Adam/80-L-BFGS runs finish below resource caps
and produce the same dense energy.

## Decision

Admit both CPU and RTX PRO 4000 Blackwell as production backends for the
restricted diagonal-path solver. Keep CPU as the automatic/default backend for
the registered small workload: its full solve takes `5.0845 s`, versus
`11.9605 s` on Blackwell. Do not claim GPU acceleration.

The historical pairwise route remains a correctness reference rather than a
production default. Exact singular pairs continue through the batched-factor
minor fallback. Future backend claims require matched hyperparameters,
value/gradient parity, and explicit time and memory measurements.

## Consequences

- The Phase 32 GPU bottleneck is resolved as a correctness and resource issue.
- Kernel vectorization gives a measured `34.926x` CPU speedup over the pairwise
  reference at the registered point.
- Device availability is separated from speed: Blackwell is valid but slower
  here, so the public claim is portability, not acceleration.
- The matched seed's direct-CI error is a reported diagnostic failure and does
  not replace the independent Phase 32 physics evidence.
- No FEMPS definition, exact antisymmetry guarantee, or scientific scope is
  changed.

## Evidence

- `docs/experiments/results/phase33_vectorized_transitions.json`
- `scripts/verify_phase33_vectorized_transitions.py`
- `docs/experiments/phase33_vectorized_transitions_report.md`
