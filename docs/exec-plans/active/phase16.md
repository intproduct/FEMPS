# Active execution plan: Phase 16 continuous ordered-distance functional basis

## Objective

Bridge the finite-grid Gate C solver back to the defining 2201 architecture:
an orthonormal local functional basis, analytic/numerical operator matrices,
native MPS/MPO contraction, and AD.  Work in center-of-mass plus positive
interparticle distances so that collision Dirichlet boundaries carry Pauli
exclusion without a finite-box charge tensor.

## Checkpoints

- [ ] Derive the exact transformation from ordered coordinates to one
  full-line center-of-mass variable and `N-1` positive distances, including the
  kinetic metric, normalization, harmonic trap, and permutation recovery.
- [ ] Implement and quadrature-test a full-line center-of-mass basis and a
  Dirichlet half-line distance basis with independent dimension and length-scale
  controls; compare finite sine-box and unbounded odd-Hermite/Laguerre options.
- [ ] Build one-variable derivative, kinetic, coordinate, and coordinate-square
  operator matrices with orthonormality, Hermiticity, boundary, and convergence
  tests.
- [ ] Express the mixed-derivative kinetic metric and harmonic quadratic form as
  exact polynomial-bond heterogeneous-site MPOs through latticeTN or a minimal
  compatible padded interface.
- [ ] Construct a separable interval-sum approximation for softened Coulomb
  interactions and report an independently measured operator/quadrature error;
  never treat a local SVD residual as a global certificate.
- [ ] Validate `N=2` and noninteracting harmonic energies against analytic
  values, then cross-check `N=4` against the finite-grid ordered oracle with
  separate basis, half-line box/scale, MPS-bond, and interaction-rank sweeps.
- [ ] Run blind multi-seed native AD optimization on RTX PRO 4000 Blackwell,
  with no product-basis state gather in training and CPU/GPU gradient parity.
- [ ] Revisit Li--Waintal and Hong et al. at the implementation level and state
  the surviving contribution no more broadly than the evidence permits.
- [ ] Issue continuous functional-basis Gate D before any larger-`N` solver
  benchmark.

## Exit criterion

Gate D passes only if the continuous-basis norm and Hamiltonian use native
polynomial MPS/MPO contraction; collision boundaries and ordered-sector
normalization are exact; basis/scale, interaction separation, and MPS-bond
errors are independently controlled; and blind `N=2` plus `N=4` results agree
with analytic or independent truth references.  A finite-grid agreement alone
does not satisfy this phase.
