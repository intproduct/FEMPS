# ADR 0012: Sparse growing-width path Gate J closes negatively

- Status: accepted
- Date: 2026-09-01

## Context

Phase 21 leaves growing virtual-state width as the weakest unclassified exact
coefficient-memory direction. Phase 22 tests a tridiagonal/fixed-bandwidth path
matrix, hoping that bounded local graph degree might support an exact transfer
recurrence beyond polynomial LC-AGP.

The upper-bidiagonal endpoint specialization has a unique virtual path and is
exactly APG. In a paired-orbital basis, its top coefficient is the permanent of
an arbitrary `M x M` 0--1 matrix. The normalized squared norm is the permanent
squared divided by `(M!)^2`, so exact squared-norm evaluation recovers the permanent.
The instance uses `D=2M`, width `M+1`, bandwidth one, and `O(M^2)` sparse input.

APG/APIG, Fischer conversion to AGP sums, and permanent-valued coefficients are
direct prior art. Ordinary Waring rank cannot be transferred through the
exterior quotient as a physical LC-AGP lower bound.

## Decision

Close Gate J negatively for generic tridiagonal and fixed-bandwidth growing-
width pair matrices. Do not build a continuous variational solver for this
candidate. Treat formal walk propagation as an arithmetic representation only,
not as a norm or Hamiltonian contraction.

## Consequences

- Fixed bandwidth and even a unique virtual path do not prevent exact
  contraction hardness.
- The sparse-path state is not a new ansatz; it intersects APG/APIG directly.
- The result is a contraction no-go, not an LC-AGP rank lower bound.
- Future exterior work must use an explicitly controlled approximation or a
  stronger statistics-carrier factorization, rather than another generic exact
  coefficient-memory subclass.
- The main project narrative remains no-go mathematics plus the independently
  validated ordered continuous control route unless a genuinely new structured
  family satisfies both novelty and contraction gates.
