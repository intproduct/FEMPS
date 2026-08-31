# Active execution plan: Phase 8 competing representations and N=8 baseline

## Objective

Add a fair ordered-sector first-quantized control and test one larger even
particle number without materializing the ordinary particle tensor. This phase
must separate exchange-statistics representation cost from optimization and
functional-basis errors before E6 soft-Coulomb physics begins.

## Checkpoints

- [ ] Specify the ordered-coordinate/Weyl-chamber Hilbert space, normalization,
  boundary conditions, and its map to a globally antisymmetric wavefunction.
- [ ] Implement a small exact ordered-sector oracle for harmonic fermions and
  cross-check its norm and energy against the exterior-sector truth.
- [ ] Define a comparison protocol for ordinary particle TT, ordered-sector
  FTN, and Pfaffian/finite-AGP FEMPS using the same one-particle basis and
  evidence levels.
- [ ] Run a noninteracting `N=8` polynomial Pfaffian/FEMPS benchmark on RTX PRO
  4000 Blackwell without constructing a `D^8` coefficient tensor.
- [ ] Run one safe interacting `N=8` point with analytic continuum energy and,
  where feasible, an independent finite exterior-sector truth.
- [ ] Report energy decomposition, antisymmetry guarantees, time, peak memory,
  basis size, variational complexity, and which representations were actually
  materialized.
- [ ] Decide, with a documented Gate B-style comparison, whether E6 should use
  finite-AGP FEMPS, ordered-sector FTN, or both as complementary controls.

## Exit criterion

One ordered-sector small-system result and one non-materialized `N=8` result
must be independently checked and reproducible. Claims must distinguish exact
structural antisymmetry, boundary-condition enforcement, basis truncation,
ansatz error, and optimizer error; no asymptotic advantage may be inferred from
a single measured point.
