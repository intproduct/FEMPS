# Completed execution plan: Phase 10 soft-Coulomb correlation hierarchy

## Objective

Establish controlled N=4 soft-Coulomb convergence in basis order D and finite-AGP
length K, then test reproducibility and one post-gate N=6 point.

## Completed checkpoints

- [x] Direct exterior Q and D truth scans through Q=128 and D=14.
- [x] No-oracle K=1--4 hierarchy at D=8 and D=10.
- [x] Three independent blind D=8,K=4 chains.
- [x] Explicit operator/basis/ansatz/optimizer error decomposition.
- [x] Safe N=6,K=2 run after N=4 stability checks.
- [x] Matched harmonic/soft-Coulomb time and memory comparison.
- [x] Electronic Pfaffian/QMC/neural-solver novelty audit update.
- [x] Decision between paper-scale expansion and solver revision.

## Exit result

Phase 10 passes. D=8 K errors fall from `1.446e-3` to `5.407e-6`; three
shorter independent K=4 chains reproduce the result at `9.19e-6`--`1.27e-5`.
The best total difference from D=14 truth drops from `2.015e-4` at D=8 to
`7.131e-5` at D=10. D=10,K=4 retains full rank but reaches overlap condition
143.5, so finite-AGP canonical conditioning is prioritized before a paper-scale
suite. See `docs/experiments/soft_coulomb_hierarchy_report.md`.
