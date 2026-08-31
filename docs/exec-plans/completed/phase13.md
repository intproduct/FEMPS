# Completed execution plan: Phase 13 beyond-LC-AGP FEMPS structure gate

## Objective

Identify a matrix-wedge exterior functional tensor-network organization beyond
an explicit polynomial-size LC-AGP calculation, or produce a precise
obstruction and pivot decision.

## Completed checkpoints

- [x] Proved the exact finite-AGP, LC-AGP, and matrix-wedge relation.
- [x] Defined and tested the minimal virtual-matrix pair-power candidate.
- [x] Derived and independently verified `N=4,chi=2` norm and one-body forms.
- [x] Defined the gauge-invariant one-body correlation multiplicity
  `N/Tr(gamma^2)`, equal to one for a single Slater.
- [x] Constructed shift-tagged Cayley-determinant data and verified the identity
  for `n=1,2,3`.
- [x] Embedded the candidate into original one-form matrix-wedge FEMPS with
  polynomial bond.
- [x] Audited LC-AGP, Gaussian MPS, noncommutative Pfaffian/quasideterminant,
  and determinant-complexity neighbors.
- [x] Issued ADR 0003 and the Phase 13 report.

## Gate B decision

`FAIL` for unrestricted dense matrix-wedge FEMPS as an exact scalable solver,
conditional on standard permanent hardness. Retain LC-AGP as a control and
pivot primary work to restricted-algebra/no-go or ordered-sector alternatives.

## Verification

- `103 passed`.
- Compile check passed.
- Whitespace check passed.
