# ADR 0011: Fixed-state growing-memory Gate I closes negatively

- Status: accepted
- Date: 2026-09-01

## Context

Phase 20 proves that uniformly bounded semisimple block size and radical depth
force matrix-pair states into polynomial-size LC-AGP. Phase 21 tests whether
growing radical depth alone can escape that collapse.

The one-generator algebra `C[z]/(z^d)` collapses by exact coefficient
interpolation to at most `M(d-1)+1` AGPs. The next candidate adds genuine
noncommutativity and two alternating path branches while retaining dimension
`2d-1`. It embeds faithfully in `Mat_2(C[z]/z^d)` and collapses to at most
`[M(d-1)+1] binom(M+3,3)` AGPs. Exact-rational certificates cover arbitrary
boundaries for both candidates.

The same argument applies whenever coefficient memory has fixed matrix-state
width over a fixed number of commuting truncated grading counters.

## Decision

Close Gate I negatively for one-generator and fixed-state graded growing-memory
coefficient algebras. Their tractability comes from explicit polynomial LC-AGP
organization and fails the beyond-LC-AGP novelty criterion.

Do not develop continuous variational solvers for these rejected candidates.

## Consequences

- Growing radical depth is necessary to evade Phase 20 but is not sufficient.
- Noncommutativity and a fixed number of path states are also insufficient.
- A remaining exact-algebra candidate must use growing state width, growing
  independent noncommutative counters, or structure outside the graded matrix
  representation, while avoiding the Phase 13 hard tagged construction.
- Border-rank limits remain approximate objects and do not substitute for an
  exact contraction theorem.
