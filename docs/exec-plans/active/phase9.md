# Active execution plan: Phase 9 E6 soft-Coulomb physics

## Objective

Move beyond the analytically solvable harmonic interaction to a one-dimensional
electronic-like soft-Coulomb pair potential while preserving independent truth,
exact antisymmetry, and a controlled functional-basis/operator approximation.

## Checkpoints

- [ ] Fix the soft-Coulomb Hamiltonian convention, units, coupling, softening
  length, spin sector, and continuum/domain boundary assumptions.
- [ ] Construct HO-basis two-body integrals with an independently converged
  quadrature and a documented factorization error.
- [ ] Cross-check the factorized polynomial AGP two-body contraction against
  explicit exterior-sector Hamiltonians for small `N,D`.
- [ ] Establish quadrature and basis convergence for `N=2` before optimizing
  finite-AGP states.
- [ ] Run blind/restarted `N=2` and `N=4` AD benchmarks, separating quadrature,
  basis, ansatz, and optimizer errors.
- [ ] Add a small ordered-grid sector control using the same soft-Coulomb
  convention and quantify its grid/box error.
- [ ] Attempt `N=6` only after the `N=4` truth and restart diagnostics pass.
- [ ] Report contraction time, memory, overlap conditioning, antisymmetry
  guarantee, and all materialization limits.

## Exit criterion

At least one interacting soft-Coulomb `N=4` point must have converged operator
construction, independent finite-sector truth, reproducible variational energy,
and polynomial/exterior agreement. No realistic-electronic or scaling claim is
allowed from the one-dimensional softened model alone.
