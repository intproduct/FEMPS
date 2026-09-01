# ADR 0027: Preregister clean-source seed robustness

- Status: accepted before either Phase 38 production lineage
- Date: 2026-09-02
- Depends on: ADR 0026 and accepted Phase 37

## Context

Phase 37 closes the historical-checkpoint dependency at one registered
N4,D6,Q128 soft-Coulomb point, but its candidate pool and optimizer seeds were
selected by a disclosed bounded exploratory audit.  One successful schedule
does not establish that a user starting from the same canonical Slater will
obtain a stable K1--K4 sequence under fresh random schedules.

This phase tests schedule robustness before changing particle number, basis
order, optimizer budget, model, contraction route, or stopping semantics.  It
is a falsifiable replication gate, not a search for a better seed.

## Decision

Run exactly two fresh complete clean-source schedules, called A and B.  Both
use the public Phase 37 command contract and retain all physical, optimizer,
resource, and validation settings.  Only the registered source-optimizer,
candidate-pool, and adaptive-optimizer seeds differ.

The machine-readable sources of truth are:

- `docs/experiments/configs/phase38_n4_d6_k4_seed_a.json`;
- `docs/experiments/configs/phase38_n4_d6_k4_seed_b.json`.

Schedule A uses source seed 3801 and K2/K3/K4 candidate/optimizer pairs
3811/3812, 3821/3822, and 3831/3832.  Schedule B uses source seed 3901 and
pairs 3911/3912, 3921/3922, and 3931/3932.  These values are frozen before
either production result is opened.

Schedule A is forced to stop after K2 and resume to K4, then is repeated once
as a clean uninterrupted control.  Schedule B is run once cleanly.  Every
registered outcome, including failure, is retained.  No seed may be replaced
and no run may be retried on the basis of its final energy.

## Frozen acceptance gates

- A-resumed, A-clean, and B-clean all complete K1--K4 without schedule change
  or hidden retry.
- A clean/resume energies and selected candidates agree within `1e-11` at
  every K.
- Every lineage is energy-nonincreasing in K within `1e-9`.
- Every K1 point has same-basis CI error at most `2e-3` and variance at most
  `1e-2`; every final K4 point has CI error at most `1e-6` and variance at most
  `1e-5`.
- Across Phase 37, A, and B, the final-energy spread is at most `2e-6`; the
  maximum absolute difference of a fresh final energy from Phase 37 is at most
  `1e-6`.
- Every stage has norm error at most `1e-10`, structural and independently
  materialized antisymmetry residual at most `1e-12`, zero virtual-path
  enumeration, and zero production `D^N` materialization.
- Operator factorization error is at most `1e-11`; each optimizer stage is at
  most 120 seconds, each command lineage is at most 600 seconds, and sampled
  peak process RSS is at most 2 GiB.
- The optimizer failure count is exactly zero.  All condition diagnostics,
  selected candidates, energy spread, CI errors, variances, times, and peak
  memory are reported whether or not the aggregate decision passes.
- An independent verifier rebuilds the Hamiltonian, exterior coefficients,
  norms, energies, variances, particle-TT ranks, candidate selections, source
  and operator identities, and all aggregate gates from committed data.

## Failure rule

Any failed lineage or gate is preserved without changing seeds, tolerances,
budgets, N, D, K, or model.  Failure classifies the Phase 37 result as schedule
sensitive and makes optimizer/initialization stability the next algorithm
gate.  It does not authorize seed replacement, automatic stopping, N/D
expansion, generic-FEMPS claims, stochastic claims, or high-dimensional
form-rank work.

## Scientific boundary

Passing establishes clean-source schedule robustness for two additional
registered schedules at one small interacting continuum point.  It does not
establish universal seed independence, asymptotic scaling, runtime
superiority, automatic stopping, or generic matrix-wedge contraction.
