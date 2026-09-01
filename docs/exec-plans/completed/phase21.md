# Completed execution plan: Phase 21 growing-memory exterior classification

## Objective

Test the weakest escapes from the Phase 20 bounded-algebra collapse: growing
radical depth with one commuting generator, then the smallest noncommutative
fixed-state path memory. Admit no numerical solver unless a candidate is both
beyond polynomial LC-AGP and exactly contractible.

## Completed checkpoints

- [x] Proved that arbitrary boundaries of `C[z]/(z^d)` pair powers require at
  most `M(d-1)+1` scalar AGPs.
- [x] Distinguished exact Waring rank from border/cactus rank and audited
  Waring monomials, Veronese jets, curvilinear schemes, moment tensors, and
  border-rank-motivated AGP-CI.
- [x] Certified every boundary basis functional exactly for all 16 cases
  `1<=M,d<=4`.
- [x] Selected the noncommutative alternating-word algebra
  `C<x,y>/(x^2,y^2,words of length d)` with dimension `2d-1`.
- [x] Proved its faithful embedding in `Mat_2(C[z]/z^d)` and exact LC-AGP bound
  `[M(d-1)+1] binom(M+3,3)`.
- [x] Certified direct word-algebra powers against the nested exact
  interpolation for every boundary word at `1<=M<=3,1<=d<=4`.
- [x] Generalized to fixed matrix-state width and a fixed number of commuting
  grading counters, and audited weighted-automata/rational-series prior art.
- [x] Issued ADR 0011 and closed Gate I negatively without continuous
  variational experiments.

## Gate I result

Gate I is **FAIL for one-generator and fixed-state graded growing memory**.
Growing radical depth, noncommutativity, and a fixed number of path branches are
jointly insufficient: exact tractability still comes from a polynomial LC-AGP
expansion. The remaining exact coefficient-algebra boundary requires growing
state width, growing independent noncommutative counters, or structure outside
the graded matrix representation, while Phase 13 excludes an unrestricted
search.
