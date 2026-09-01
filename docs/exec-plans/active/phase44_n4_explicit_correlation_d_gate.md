# Active execution plan: Phase 44 interacting N=4 explicit-correlation D gate

## Objective

Execute ADR 0033 exactly once to determine whether the validated continuous
coordinate backend gives a non-NOCI carrier-basis `D`-convergence advantage on
the interacting four-fermion soft-Coulomb model. This is a falsifiable physics
gate, not a Paper B drafting phase.

## Work packages

1. Finalize the checkpointed Adam/QR stochastic optimizer and keep its state,
   RNG, proposal counts, histories, and source identities exactly resumable.
2. Implement a production runner that extracts only the disclosed historical
   Phase 37 `K=1` carrier and enforces the reference firewall.
3. Run the six frozen `D/seed` optimizations, including the forced D6 resume
   and clean-trajectory comparison.
4. Perform frozen selection evaluations, serialize/hash the three choices,
   and only then admit the fixed historical CI/NOCI/D14 comparator values.
5. Run the six held-out confirmation evaluations and archive raw coordinates.
6. Independently reconstruct every selected state and recompute all
   observables, uncertainty diagnostics, symmetry residuals, point gates, and
   aggregate decisions from committed raw data.
7. Report the complete passing or failing outcome, resource measurements, and
   the scientific boundary. Do not add a rescue point.

## Immediate implementation gates

- small-system materialization/local-energy identities remain green;
- stochastic optimizer interruption/resume equals a clean trajectory exactly;
- changed initialization, exponent, configuration, or checkpoint identity is
  rejected;
- no production code path imports or reads reference energies before the
  serialized selection boundary;
- all output paths and hashes are platform independent;
- production never forms a `D^N` coefficient tensor, full alternating tensor,
  or virtual-path list.

## Stop conditions

Stop and retain the result if any frozen budget, convergence diagnostic,
symmetry threshold, resource limit, resume identity, or differentiator gate
fails. Do not broaden `D`, change the correlator, run additional NOCI controls,
or edit either manuscript in response to the result.

## Deliverables

- production and verification scripts;
- immutable optimization checkpoints and hashed coordinate archives;
- machine-readable result with the reference-firewall chronology;
- a concise Phase 44 experiment report;
- focused tests and the standard repository suite;
- a remote Git checkpoint when the phase is complete.
