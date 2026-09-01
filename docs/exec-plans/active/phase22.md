# Active execution plan: Phase 22 sparse growing-width path gate

## Objective

Test the next necessary escape from the Phase 21 collapse: matrix-state width
that grows with the problem, but with a sparse path-graph generator rather than
unrestricted dense matrices. Determine representational separation from
polynomial LC-AGP and exact contraction complexity before any continuous
variational implementation.

## Candidate J1

Use an open-boundary tridiagonal or fixed-bandwidth `w x w` matrix of scalar
physical two-forms, with independent diagonal/edge forms and selected endpoint
boundaries. Its M-th power is a weighted length-M walk polynomial on a path.
The number of matrix states grows, but local graph degree stays bounded.

## Checkpoints

- [ ] State the precise coefficient family, gauge redundancies, boundary class,
  and joint input parameters `(M,D,w)`.
- [ ] Derive exact walk-polynomial recurrences without claiming that amplitude
  evaluation implies norm or Hamiltonian contraction.
- [ ] Test polynomial LC-AGP separation using catalecticant/Waring lower bounds
  on exact symbolic small instances; distinguish exact, border, and approximate
  rank.
- [ ] Attempt an exact norm/one-body/factorized-two-body recurrence and give
  explicit time and memory costs in `(M,D,w)`.
- [ ] Test whether Phase 13 tagged determinant hardness survives tridiagonal or
  fixed-bandwidth virtual matrices; stop solver work if a reduction survives.
- [ ] Audit continuant/walk polynomials, algebraic branching programs, sparse
  matrix powers, weighted automata with growing state count, and structured
  Waring rank prior art.
- [ ] Build only bounded exact certificates and exterior truth checks until the
  contraction and novelty gates both resolve.
- [ ] Issue Gate J. No GPU optimization is allowed from favorable amplitude or
  parameter-count scaling alone.

## Exit criterion

Gate J requires both a proof that the family is not polynomial-size LC-AGP/
Gaussian in its admitted parameters and an exact polynomial recurrence for norm
and functional one-/two-body operators. A conditional hardness result or a new
LC-AGP collapse is an acceptable negative outcome. Sparse amplitude propagation
by itself is insufficient.
