# Completed execution plan: Phase 4 structured functional solver

## Objective

Reconnect the conditionally admitted fixed-number Pfaffian FEMPS subclass to
the 2201 functional-basis Hamiltonian and pass the E1/E2 continuum benchmarks.

## Completed checkpoints

- [x] Add stabilized/scaled overlap recurrences for optimization trajectories.
- [x] Implement blocked Pfaffian states for odd particle numbers.
- [x] Define functional one- and two-body operator containers with explicit
  operator-Schmidt rank and dtype/device metadata.
- [x] Build normalized AGP and finite-AGP-sum energy functionals.
- [x] Reuse latticeTN optimizer, device, checkpoint, and resume conventions.
- [x] E1: reproduce two noninteracting spinless harmonic fermions with exact
  antisymmetry, energy, gradient, and pair rank one.
- [x] E2: reproduce the analytically separable interacting harmonic pair.
- [x] Scan basis order and Pfaffian/AGP expressivity controls with raw JSON.
- [x] Compare against full antisymmetric truth and ordinary particle-TT ranks.

## Exit result

E1 and E2 reach documented high precision with full-reference value and
gradient agreement, stable resume, and no unreported symmetry or contraction
approximation. The stability and single-block odd-particle extensions also pass
independent truth and Blackwell parity checks.
