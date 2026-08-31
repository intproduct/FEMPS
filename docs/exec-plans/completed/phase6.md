# Completed execution plan: Phase 6 finite-AGP solver conditioning

## Objective

Turn the E4 finite-AGP representation capacity into a reliable blind
variational algorithm before increasing particle number.

## Completed checkpoints

- [x] Expose polynomial overlap and Hamiltonian transition matrices with
  Hermiticity and explicit-truth tests.
- [x] Solve amplitudes by a conditioned generalized Hermitian eigenproblem.
- [x] Add pair scale/phase gauges, deterministic output ordering, and overlap
  near-dependence control.
- [x] Add deterministic checkpoint/resume and best-state restoration.
- [x] Replace failed simultaneous random K=2 optimization by no-oracle greedy
  K=1-to-K=2 growth and joint relaxation.
- [x] Obtain three reproducible `D=8,kappa=0.35` errors of
  `2.00e-5`--`3.11e-5` with stable overlap diagnostics.
- [x] Repeat the finite-AGP hierarchy at `D=10` and place its representation
  error below the oscillator-basis error.

## Exit result

Phase 6 passes its documented tolerance. Polynomial/exterior agreement,
effective overlap rank, condition number, generalized residual, and restart
behavior are all explicitly verified. E5 may proceed with greedy growth as the
default finite-AGP solver strategy.
