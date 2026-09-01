# Active execution plan: Phase 35 Adaptive-Pool Stability and Stop Calibration

## Objective

Determine whether the Phase 34 truth-free K-growth result is stable across
independent seeded candidate pools at the same interacting `N=6,D=12` point,
and assess whether fixed-span predicted improvements provide a defensible
automatic stopping signal. Do not expand particle number or basis order.

## Preregistered workload to freeze in ADR 0024

- Start every lineage from the same accepted Phase 32 D12,K4 checkpoint.
- Use 32 candidates per growth step and three fresh seed pairs:
  `(3511,3512)`, `(3521,3522)`, and `(3531,3532)` for K5/K6.
- Use the unchanged 160 Adam plus 80 L-BFGS optimizer schedule on CPU.
- Complete all six optimizations before constructing dense CI or reading the
  Phase 34 final errors.
- Reuse the Phase 34 cold K6 only as a named historical control; do not add
  seed-specific rescue starts.

## Primary work

1. Add ADR 0024 and a source-hashed multiseed runner before production.
2. Run and checkpoint all three K4-to-K5-to-K6 lineages.
3. Independently reconstruct energies, variances, norms, TT ranks, storage,
   conditions, time, memory, and antisymmetry records from exterior
   coefficients.
4. Compare fixed-span predicted improvements with fully reoptimized gains.
   Admit an automatic stopping rule only if one preregistered rule is consistent
   across all lineages; otherwise report that stopping remains externally
   capped.
5. Update the method evidence only after the multiseed gate closes.

## Acceptance gates

- Every lineage has nonincreasing K4/K5/K6 energy within `1e-9` and total
  K4-to-K6 improvement of at least `1e-8`.
- Every K6 error versus same-basis CI is at most `1.1e-4`, variance at most
  `1.5e-3`, and the three K6 energies have spread at most `1e-4`.
- Norm error is at most `1e-10`, structural antisymmetry residual at most
  `1e-12`, balanced condition at most `1e8`, and all K directions survive.
- Production enumerates zero virtual paths and materializes zero `D^N` particle
  tensors. Every point stays below 600 seconds and 2 GiB sampled CPU RSS.
- Operator factorization, quadrature, factorized/dense energy, source lineage,
  deterministic seed, and source-hash gates remain unchanged.
- A committed artifact and independent verifier reproduce every accepted
  numerical statement; full tests and manuscript evidence lint pass.

## Boundaries and failure rule

- No N8, D expansion, high-dimensional form-rank search, extra pool seed, or
  post-result optimizer change.
- CI data cannot influence selection, optimization, stopping, or thresholds.
- A failed lineage is reported as adaptive-pool instability. It cannot be
  replaced by the best two of three.
- Failure of a common stopping rule leaves K externally capped; it does not
  authorize an unsupported efficiency claim.
