# ADR 0010: Bounded coefficient-algebra Gate H closes negatively

- Status: accepted
- Date: 2026-09-01

## Context

Phase 20 asks whether a restricted matrix-wedge coefficient algebra can go
strictly beyond finite LC-AGP/Gaussian families while retaining exact
contraction polynomial jointly in particle number, basis size, and virtual
dimension. The first candidate uses the noncommutative upper-triangular algebra
`T_2`; its square-zero radical yields an exact expansion in at most
`binom(M+2,2)+2` scalar AGPs.

The broader classification resolves a finite-dimensional complex coefficient
algebra into semisimple matrix blocks and its Jacobson radical. If the largest
matrix block has fixed size `p` and the radical has fixed nilpotency index `d`,
only fewer than `d` radical insertions survive. After resolving block paths and
radical basis elements, every contribution is a homogeneous polynomial in a
constant number of commuting physical two-forms and hence a polynomial-size
sum of M-th powers. Those powers are scalar fixed-number AGPs.

The fully noncommutative semisimple base case `Mat_2` needs at most
`binom(M+3,3)` AGPs. Independent exact-rational certificates cover `T_2` for
M=1--6 and a generic symbolic `Mat_2` matrix with deterministic nonzero
boundaries for M=1--4. The general proof remains labeled a theorem draft until
external review.

## Decision

Close Gate H negatively for coefficient-algebra families whose largest
semisimple matrix block and radical nilpotency index are uniformly bounded.
Their exact tractability is inherited from a polynomial-size LC-AGP expansion,
so they do not satisfy the predeclared beyond-LC-AGP novelty criterion.

Do not launch variational N=4/N=6 studies for this rejected family. Retain the
implementations and exact certificates as algebraic regression oracles.

## Consequences

- Noncommutativity alone is not a sufficient correlation resource for
  symmetrized matrix-valued pair powers.
- Within exact finite-dimensional coefficient algebras, escaping this theorem
  requires a simple-block size or radical nilpotency depth that grows with the
  problem. This is necessary but does not guarantee tractability or novelty.
- The Phase 13 hard shift tags sit on the growing-radical side of this boundary.
- The next research branch must either find extra contractible structure in a
  growing family, prove a stronger negative classification, or formulate an
  approximate contraction with a quantitative error certificate.
