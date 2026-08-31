# ADR 0002: Gate A conditional pass through fixed-number Pfaffians

## Status

Accepted on 2026-08-31.

## Decision

Gate A is **CONDITIONAL**.

- Unrestricted matrix-wedge FEMPS is retained as a mathematical ansatz and
  small-system oracle; no generic polynomial contraction is claimed.
- Phase 4 numerical solver work is authorized for fixed-number Pfaffian/AGP
  FEMPS and finite AGP sums.
- General two-body operators must enter through an explicit operator-Schmidt,
  density-fitting, or other separable factorization with recorded rank `L`.
- Pair rank, AGP count, antisymmetry residual, and all contraction costs must be
  reported independently.

## Evidence

The structured state has bond `r`, contains `binom(r,M)` ordered Slater paths,
and is non-decomposable for generic `M>1,r>M`. Pfaffian generating functions
contract its overlap in `O(MD^3)` time and `O(D^2+M)` memory. One-/two-body
formulas, finite sums, AD gradients, and Blackwell parity have independent
explicit-tensor validation.

## Consequences

This decision does not establish novelty or advantage over established AGP,
number-projected BCS, or Pfaffian methods. The project's remaining potential
contribution is their precise embedding as a first-quantized 2201-compatible
functional exterior TN, together with AD functional operators and a controlled
comparison against ordinary particle TT, Slater sums, and ordered sectors.

Odd-particle blocked Pfaffians, stabilized recurrences, checkpointed
optimization, and E1/E2 benchmarks are Phase 4 requirements.
