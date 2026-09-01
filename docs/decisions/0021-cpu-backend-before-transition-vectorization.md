# ADR 0021: Use CPU production until determinant transitions are vectorized

- Status: accepted
- Date: 2026-09-01
- Depends on: ADR 0017, ADR 0019, Phase 32 resource preflight

## Context

The restricted diagonal-path FEMPS contraction is polynomial, but the current
reference implementation loops in Python over Slater transition pairs and
physical operator-SVD factors. A Blackwell complex128 determinant/AD probe
passes, so device capability is not the obstacle. On the registered
`N=6,D=10,K=4`, 160-Adam/80-LBFGS workload, however, the GPU branch did not
finish within the 600 s resource limit and was stopped. A contemporaneous
hardware snapshot showed only partial GPU utilization and about 1.34 GiB total
device use, consistent with many small kernel launches rather than memory
pressure.

The same registered CPU workload completes in 88.1 s. The larger
`N=6,D=12,K=4` CPU continuation completes in 107.7 s with 974,974,976 sampled
peak RSS bytes, below the written 244.4 s and 1.92 GB preflight estimate.

## Decision

Use CPU as the Phase 32 production backend. Preserve the stopped Blackwell run
only as backend diagnostic evidence; it is not a scientific convergence point.
Do not describe GPU availability as acceleration. Phase 33 may vectorize the
transition/factor axes, but must prove value and gradient parity before
repeating a matched backend benchmark.

## Consequences

- Phase 32 D/K evidence is backend-stable: completed CPU and diagnostic GPU
  K1/K2 energies agreed to approximately `1e-14` before the K4 GPU stop.
- Current performance claims are CPU-local and bounded to the registered points.
- No algorithm definition, FEMPS carrier, contraction formula, or physics model
  changes.
- A future GPU route is admitted only by the matched Phase 33 gate; changing
  budgets or workloads to manufacture a speedup is forbidden.

## Evidence

- `docs/experiments/results/phase32_n6_resource_audit.json`
- `docs/experiments/results/phase32_n6_convergence.json`
- `scripts/verify_phase32_n6_resource_audit.py`
- `scripts/verify_phase32_n6_convergence.py`
