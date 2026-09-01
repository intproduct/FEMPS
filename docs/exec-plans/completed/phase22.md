# Completed execution plan: Phase 22 sparse growing-width path gate

## Objective

Test the next necessary escape from the Phase 21 collapse: matrix-state width
that grows with the problem, but with a sparse path-graph generator rather than
unrestricted dense matrices. Determine representational separation from
polynomial LC-AGP and exact contraction complexity before any continuous
variational implementation.

## Completed checkpoints

- [x] Defined the upper-bidiagonal endpoint family with `w=M+1`, its diagonal
  similarity gauge, `D=2M` hard specialization, and `O(M^2)` binary input.
- [x] Derived the exact exterior walk recurrence and separated compact formal
  propagation from physical coefficient and norm extraction.
- [x] Identified the unique-path state exactly as APG and corrected the proposed
  ordinary-Waring lower bound: the exterior quotient prevents importing it as
  a physical LC-AGP rank theorem.
- [x] Audited APG/APIG, Fischer-to-AGP decompositions, generic geminal RDM
  intractability, and permanent-valued determinant coefficients.
- [x] Proved that paired-orbital edge forms encode `perm(A)` in the top-form
  coefficient and `perm(A)^2/(M!)^2` in the normalized squared norm.
- [x] Showed that the reduction has one virtual path and bandwidth one, so every
  broader tridiagonal/fixed-bandwidth family inherits the exact obstruction.
- [x] Added an implementation-independent exact certificate comparing virtual-
  path, exterior-subset, and permutation routes for `1<=M<=6`.
- [x] Issued ADR 0012 and closed Gate J negatively without GPU or variational
  solver development.

## Gate J result

Gate J is **FAIL for generic growing-width path and fixed-bandwidth pair
matrices**. The weakest upper-bidiagonal candidate is established APG/APIG,
while exact norm contraction recovers the #P-complete 0--1 permanent. Sparse
virtual degree, a unique path, and polynomial parameter count therefore do not
supply a tractable exterior state family.

The result is not an LC-AGP rank lower bound: at `D=2M` the hard output lies in
a one-dimensional top exterior sector. It is a coefficient-evaluation and norm-
contraction obstruction.
