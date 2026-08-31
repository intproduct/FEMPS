# Active execution plan: Phase 10 soft-Coulomb correlation hierarchy

## Objective

Turn the first E6 points into a controlled `D,K,N` hierarchy, quantify the
remaining finite-AGP representation error, and decide whether the current
subclass is sufficient for a first FEMPS science result.

## Checkpoints

- [ ] Sweep N=4 basis order and quadrature order with independent exterior truth.
- [ ] Extend no-oracle greedy growth through K=4 and test overlap-rank stability.
- [ ] Repeat the best N=4 configuration over at least three blind seeds.
- [ ] Separate operator, basis, finite-AGP, and optimizer errors in one table.
- [ ] Run an N=6 K=2 point only after the N=4 hierarchy is stable.
- [ ] Compare contraction time and peak memory against harmonic E4/E5 at matched
  `N,D,K`, explicitly accounting for soft-Coulomb factor rank.
- [ ] Update the novelty audit for determinant/Pfaffian soft-Coulomb solvers.
- [ ] Decide whether to prepare a paper-scale benchmark suite or revise the
  structured ansatz/optimizer first.

## Exit criterion

The N=4 soft-Coulomb energy must show reproducible convergence in both `D` and
`K`, with every reported production point independently checked in the exterior
sector and with stable generalized-eigen diagnostics. No N-scaling claim may be
made before this two-axis convergence is established.
