# Completed execution plan: Phase 15 ordered-distance functional-TN gate

## Completed checkpoints

- [x] Fixed the finite-grid gap coordinates, collision/hard-wall boundaries,
  local cutoff, normalization, and bijection with strictly ordered particle
  configurations.
- [x] Derived and tested the adjacent-gap-transfer kinetic operator against the
  independently assembled ordered-coordinate Hamiltonian.
- [x] Constructed exact polynomial-bond MPOs for the cumulative-coordinate
  harmonic trap and the finite-box total-gap projector.
- [x] Constructed exact fixed-charge interval-counter MPOs for every
  soft-Coulomb particle pair and measured optional compression error
  independently.
- [x] Connected the gap MPOs to latticeTN native energy and AD without a
  `d**(N+1)` production gather.
- [x] Separated local gap cutoff, MPS bond, MPO operator, grid, and box controls
  on the `N=4,L=8` ordered truth problem.
- [x] Added a hard cumulative-charge variational MPS and reproduced the truth
  from three random Blackwell initializations.
- [x] Compared the scope with Li--Waintal and restricted the possible project
  contribution to the 2201 continuous functional-basis bridge plus no-go
  theory.
- [x] Issued ADR 0005: Gate C passes at finite-grid scope.

## Principal evidence

- Exact finite-grid truth energy: `10.550426086401501`.
- Native exact-MPS/MPO error: `5.33e-15`.
- Exact gap-MPS internal ranks: `(5,10,10,5)`.
- Raw Hamiltonian MPO maximum bond: `33`, with general bound `O(N^2(L-N))`.
- Three blind 2500-step energy errors:
  `8.33e-6`, `2.07e-5`, and `1.30e-6`.
- Three post-training fidelities exceed `0.9999956`; charge weight is exactly
  one and forbidden parameters remain exactly zero.
- CPU/RTX PRO 4000 Blackwell energy and gradient maximum differences are both
  `5.33e-15`.
- Training materializes no product-basis state; dense tensors occur only in
  explicitly small truth/operator audits.

## Verification

- Full suite: `119 passed`.
- Ordered-distance focused suite: `12 passed`.
- Python compile and Git whitespace checks passed.
- Exact tagged-Cayley certificate reverified with hash
  `893077be401414cd810fa1154e618d37d83b58e077732801f2482b3716b2c0c0`.

## Decision and limitation

Gate C is **PASS (finite-grid scope)**.  This establishes a controlled
polynomial native representation and a functioning variational solver, not a
continuum or large-`N` result.  Phase 16 must restore the 2201 orthonormal
functional-basis layer with full-line center-of-mass and Dirichlet half-line
distance variables before the solver branch can advance.
