# Active execution plan: Phase 11 finite-AGP canonical conditioning

## Objective

Control near-dependence and seed sensitivity in finite-AGP sums so that larger
basis orders and K can be optimized without relying on accidental redundant
directions or ill-conditioned amplitude solves.

## Checkpoints

- [ ] Diagnose whether D=10 overlap conditioning comes from pair similarity,
  amplitude cancellation, scale/phase gauge, or optimizer trajectory.
- [ ] Define an amplitude-aware, gauge-invariant finite-AGP correlation spectrum
  without calling it an entanglement spectrum.
- [ ] Implement deterministic overlap whitening/compression with explicit state
  and energy preservation tests.
- [ ] Add a safe rule for pruning or restarting near-dependent nonlinear terms.
- [ ] Re-run D=10,K=4 over at least three seeds and compare condition, energy,
  retained rank, and optimizer spread against Phase 10.
- [ ] Test whether K>4 supplies reproducible improvement after conditioning.
- [ ] Verify all compression decisions against explicit exterior states at safe D.
- [ ] Decide whether the conditioned representation is ready for paper-scale
  benchmarking or requires a different structured ansatz.

## Exit criterion

A conditioned D=10,K=4 workflow must reproduce the Phase 10 energy or improve
it over three seeds while reducing pathological overlap sensitivity, preserving
the exterior state/energy within tolerance, and reporting every discarded or
restarted direction explicitly.
