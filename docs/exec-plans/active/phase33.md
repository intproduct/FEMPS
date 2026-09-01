# Active execution plan: Phase 33 N6 Evidence Integration and Transition Vectorization

## Objective

Integrate the independently verified Phase 32 N6 D/K convergence into the
restricted-method manuscript and remove the measured small-kernel backend
bottleneck without changing the first-quantized continuous FEMPS definition or
weakening any exact-contraction and antisymmetry guarantees.

## Primary work

1. Add the Phase 32 D/K tables, operator/quadrature audit, variance, symmetry,
   resource, Slater/CI/ordinary-TT comparisons, and limitations to the method
   manuscript and reproduction manifest.
2. Vectorize transition pairs and the physical operator-SVD factor axis in the
   diagonal-path contraction while retaining the singular-safe minor fallback.
3. Require value and reverse-mode gradient parity against the current reference
   implementation on CPU before any performance claim.
4. Benchmark matched `N=6,D=10,K=4` CPU and Blackwell runs. Blackwell becomes a
   production backend only if it completes within 600 s, stays below 4 GiB,
   and reproduces energy/gradient/conditioning within registered tolerances.
5. Preserve the exterior-sector ordinary-TT comparator so no benchmark needs a
   `D^N` tensor solely to report TT ranks.

## Boundaries

- No N8 expansion, high-dimensional alternating-form rank search, or generic
  exact-contraction claim.
- No change of FEMPS into an occupation-number or second-quantized MPS.
- Optimization hyperparameters may not be changed inside a backend comparison.
- A failed speedup is reported as a backend limitation, not hidden by replacing
  the matched workload.
- Direct CI remains the named same-basis truth and is not an initializer.

## Acceptance gates

- Phase 32 artifact and independent verifier remain unchanged and passing.
- Vectorized values agree with the reference to `1e-11`; reverse-mode gradients
  agree to `1e-9` on registered well-conditioned and fallback cases.
- Every production result reports structural antisymmetry residual, time,
  sampled RSS, device memory when applicable, and conditioning.
- Manuscript claims remain explicitly restricted, numerical, and free of
  asymptotic/scalability/superiority language not supported by the benchmark.
- Full tests, manuscript evidence lint, and release-manifest verification pass.
