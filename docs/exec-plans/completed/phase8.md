# Completed execution plan: Phase 8 competing representations and N=8 baseline

## Objective

Add a fair ordered-sector first-quantized control and test one larger even
particle number without materializing the ordinary particle tensor before E6.

## Completed checkpoints

- [x] Define the normalized Weyl-chamber isometry, collision boundary, and
  signed extension to the full antisymmetric wavefunction.
- [x] Implement exact tensor-map and local harmonic-grid ordered-sector oracles.
- [x] Cross-check four `N=3` ordered Hamiltonians entrywise against independent
  exterior Slater--Condon truth.
- [x] Define an evidence-aware ordinary-TT / ordered-sector / FEMPS protocol.
- [x] Complete noninteracting and interacting `N=8,D=10` polynomial Blackwell
  runs without materializing the `D^8` particle tensor.
- [x] Record energy decomposition, fidelity, wall time, peak memory, and exact
  materialization scope.
- [x] Issue the Gate B competing-representation decision.

## Exit result

Phase 8 passes. Noninteracting E8a reaches energy `32`; interacting E8b at
`kappa=0.02` reaches `3.709e-6` finite-basis error with polynomial/exterior
agreement at `7.82e-14`. The ordered-sector oracle is exact for local grid
Hamiltonians but is not yet a production functional MPS solver. E6 proceeds
with finite-AGP FEMPS as the main route and ordered sectors as complementary
small-system controls. See `docs/experiments/phase8_scaling_report.md`.
