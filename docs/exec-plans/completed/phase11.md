# Completed execution plan: Phase 11 finite-AGP canonical conditioning

## Objective

Control apparent near-dependence and seed sensitivity in finite-AGP sums before
larger production scans.

## Completed checkpoints

- [x] Diagnosed D10 conditioning as scale-gauge norm imbalance rather than
  pair or many-body-state near-dependence.
- [x] Defined the amplitude-aware, gauge-invariant contribution Gram spectrum
  and explicitly separated it from entanglement spectra.
- [x] Implemented deterministic diagonal balancing, whitening, and thresholded
  span compression with explicit exterior state tests.
- [x] Added an auditable leave-one-out safe-pruning rule and a duplicate-state
  exterior fidelity test.
- [x] Re-ran D10,K4 over seeds 301--303; every run preserved or improved the
  Phase 10 error with balanced condition numbers 1.75--3.06.
- [x] Grew K4 to K5 on all three chains; every run improved, with errors
  6.00e-6--8.82e-6 and balanced conditions below 3.11.
- [x] Verified polynomial/exterior energies to at worst 2.06e-13 at K4 and
  8.35e-14 at K5.
- [x] Recorded zero discarded, pruned, or restarted directions in every
  production run.

## Decision

The fixed-number finite-AGP subclass is conditioned well enough for broader
benchmarking.  Generic matrix-wedge FEMPS remains under the existing
CONDITIONAL Gate A restriction.
