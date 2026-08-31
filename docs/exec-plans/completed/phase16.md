# Completed execution plan: Phase 16 continuous ordered-distance functional basis

## Objective

Bridge the finite-grid Gate C solver back to the defining 2201 architecture:
an orthonormal local functional basis, analytic/numerical operator matrices,
native MPS/MPO contraction, and AD. Work in center-of-mass plus positive
interparticle distances so that collision Dirichlet boundaries carry Pauli
exclusion without a finite-box charge tensor.

## Checkpoints

- [x] Derived the exact transformation from ordered coordinates to one
  full-line center-of-mass variable and `N-1` positive distances, including the
  kinetic metric, normalization, harmonic trap, and permutation recovery.
- [x] Implemented and quadrature-tested a full-line center-of-mass basis and
  Dirichlet half-line distance bases with independent dimension and scale
  controls; compared finite sine-box and unbounded odd-Hermite options.
- [x] Built one-variable derivative, kinetic, coordinate, and
  coordinate-square matrices with orthonormality, Hermiticity, boundary, and
  convergence tests.
- [x] Expressed the mixed-derivative kinetic metric and harmonic quadratic form
  as exact polynomial-bond heterogeneous-site MPOs through latticeTN.
- [x] Constructed a separable interval-polynomial approximation for softened
  Coulomb interactions and reported independent scalar, projected-operator,
  quadrature, and energy errors.
- [x] Validated N=2 and noninteracting harmonic energies against independent or
  analytic values, then cross-checked N=4 against an independent exterior
  numerical reference with separate basis, half-line box/scale, MPS-bond, and
  interaction-rank sweeps. The finite-grid oracle remains a structural rather
  than continuum truth check.
- [x] Ran blind multi-seed native AD optimization on RTX PRO 4000 Blackwell,
  with no product-basis state gather in training and CPU/GPU gradient parity.
- [x] Revisited Li--Waintal and Hong et al. at implementation level and limited
  the surviving contribution to the continuous functional-basis/operator/AD
  integration and its controlled evidence.
- [x] Issued ADR 0006 and continuous functional-basis Gate D before any
  larger-N solver benchmark.

## Gate D result

Gate D is **PASS (controlled continuous-basis prototype)**. The coordinate
map, collision boundary, chamber normalization, and signed fermion recovery
are exact. Native MPS/MPO contraction handles the norm and Hamiltonian without
a product-state gather in training. Basis/scale, interaction degree,
quadrature, MPS bond, and optimization errors are independently controlled.

All three blind N=2 runs lie within `3.01e-6` of their post-training Galerkin
truth. All three blind N=4 runs lie within `4.39e-3` of an independent exterior
`D=14` numerical reference, below the declared `6e-3` threshold; their solver
errors relative to the same-basis Galerkin truth are below `4.64e-5`.

The result has controlled `N<=4` scope. The exterior reference is not a
continuum bound, the odd-Hermite basis is not yet supported by the interacting
MPO, and no larger-N accuracy or method-priority claim is inferred.
