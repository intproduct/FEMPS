# Completed execution plan: Phase 9 E6 soft-Coulomb physics

## Objective

Move beyond harmonic interactions to a controlled one-dimensional
electronic-like soft-Coulomb benchmark with exact antisymmetry and independent
operator and finite-sector truth.

## Completed checkpoints

- [x] Fix the full-line spin-polarized Hamiltonian and units.
- [x] Converge Gauss--Hermite two-body integrals and symmetric kernel factors.
- [x] Cross-check direct tensors, factorized polynomial contractions, gradients,
  and explicit exterior Hamiltonians.
- [x] Establish N=2 quadrature, basis, relative-grid, and box convergence.
- [x] Complete blind/restarted N=2 and N=4 Blackwell benchmarks.
- [x] Batch the factor axis of the mixed-derivative overlap recurrence.
- [x] Improve N=4 with no-oracle greedy K=2 growth.
- [x] Attempt one safe N=6 point and report time, memory, and truth diagnostics.

## Exit result

Phase 9 passes. Q96/Q128 direct tensors agree to `2.25e-11`; the N=2 HO and
independent grid references agree to `8.49e-8`. N=4 greedy K=2 reaches
`7.445e-5` finite-basis error with polynomial/exterior agreement at `1.39e-13`.
The batched recurrence improves matched N=2 step time by 19.9x. See
`docs/experiments/soft_coulomb_operator_report.md`.
