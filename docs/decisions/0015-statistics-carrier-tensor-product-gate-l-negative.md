# ADR 0015: Reject the universal statistics-carrier tensor product

- Status: accepted
- Date: 2026-09-01

## Context

After generic exact and relative matrix-wedge contraction failed, the remaining
master-plan exterior hypothesis was a factorization of each particle-cut space
into a fixed fermionic statistics carrier and a smaller correlation
multiplicity, with multiplicity one for every Slater determinant.

Symmetry-adapted tensor networks provide a genuine structural/degeneracy split,
so this analogy had to be tested rather than dismissed. Candidate L1 made the
object explicit as the image of the exterior contraction map and required an
exact direct tensor product, canonicality up to unitary equivalence,
symmetry-preserving truncation, and compact-input contraction.

## Decision

Gate L is **FAIL** for the universal direct tensor-product proposal.

At the `1|N-1` cut, Slater multiplicity one forces the carrier dimension to be
`N`. For every `N>=3`, the two-Slater form

```text
e_1 wedge (e_2 wedge e_3 + e_4 wedge e_5)
    wedge e_6 wedge ... wedge e_(N+2)
```

has exact cut rank `N+2`, which is not divisible by `N`. This is full support,
invariant under orbital coordinates, and stable on an open neighborhood.

The symmetry analogy does not change the dimension result. Particle
permutations provide a one-dimensional sign irrep, while `GL(V)` provides the
full irreducible exterior representation with multiplicity-one coupling. A
state-adaptive occupied subspace works only for a Slater; a selected
Slater/secant decomposition is an established determinant expansion and is
not a canonical free tensor factor.

Finally, the Phase 22 compact path is projectively one Slater while its scalar
coefficient is a permanent. Therefore a state-intrinsic multiplicity-one label
does not imply an efficient compiler, norm, or reconstruction algorithm from
FEMPS cores.

## Consequences

- Do not describe the existing contribution-Gram diagnostic as a canonical
  FEMPS bond or entanglement spectrum; its proven invariance is narrower.
- Keep symmetry-adapted charge/multiplet TNs, Hamiltonian-specific orbital
  symmetry, and finite Slater/AGP sums as valid prior-art methods.
- Do not pursue a new optimizer for Candidate L1.
- A future categorical carrier must differ explicitly from a direct tensor
  product and prove exactness, reconstruction, truncation error, and
  contraction independently.
- Consolidate the representation, exact contraction, approximation, and
  carrier results into an internally reviewable manuscript before opening
  further method branches.
