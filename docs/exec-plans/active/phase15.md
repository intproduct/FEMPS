# Active execution plan: Phase 15 ordered-distance functional-TN gate

## Objective

Replace the dense ordered-sector truth path with a first-quantized
interparticle-distance functional MPS/MPO prototype that retains the 2201
operator and AD workflow. Determine whether kinetic, trap, interaction, and
finite-domain constraints admit controlled polynomial contractions.

## Checkpoints

- [ ] Fix finite-grid distance coordinates, boundary conditions, normalization,
  and the exact map to strictly ordered coordinate configurations.
- [ ] Derive and test the nearest-neighbor mixed-derivative kinetic operator in
  distance variables against the ordered-coordinate truth Hamiltonian.
- [ ] Construct polynomial-bond MPOs for cumulative-coordinate harmonic trap
  terms and the finite-box sum constraint.
- [ ] Audit soft-Coulomb interval-sum interactions for exact or controlled
  low-rank MPO approximation, with separate operator error.
- [ ] Connect the distance-coordinate MPOs to latticeTN native energy and AD;
  prohibit dense `D**N` gathering in the production path.
- [ ] Reproduce the smallest `N=4` ordered truth energy with independent
  distance cutoff, local basis/grid, MPO bond, and MPS bond controls.
- [ ] Compare against Li--Waintal's ordered first-quantized MPS and state the
  remaining 2201 functional-basis contribution narrowly.
- [ ] Issue the ordered-distance Gate C decision before larger benchmarks.

## Exit criterion

Norm and Hamiltonian energy use native MPS/MPO contraction with an explicit
polynomial complexity bound; exact or independently bounded operator error is
reported; and the `N=4` truth comparison separates distance discretization,
constraint, MPO, and MPS errors. Otherwise the numerical solver route stops.
