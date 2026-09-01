# Phase 25 report: statistics-carrier/multiplicity Gate L

## Outcome

The literal master-plan factorization

```text
B_k(C) ~= S_(N,k)^fermion tensor M_k(C)
```

cannot hold for every exterior state while assigning multiplicity one to a
Slater. Gate L closes negatively before numerical optimization.

## Dimension theorem

A Slater has `r_1=N`, so its multiplicity-one condition fixes the one-cut
carrier dimension to `N`. The exact two-Slater family

```text
C_N = e_1 wedge (e_2 wedge e_3 + e_4 wedge e_5)
      wedge e_6 wedge ... wedge e_(N+2)
```

has `r_1=N+2`. Since this is not a multiple of `N` for `N>=3`, a free direct
tensor product is impossible.

The two component contraction images would span `2N` dimensions if supplied
independent covectors. The physical sum supplies one shared covector, locking
`N-2` common-orbital channels and leaving `N+2`. This is a state-dependent
linear relation, not an independent correlation multiplicity index.

## Canonicality and stability audit

- `S_N` antisymmetry supplies only the one-dimensional sign representation;
  `binom(N,k)` is not a particle-permutation irrep multiplicity.
- `GL(V)` makes `Lambda^N V` irreducible, and its exterior cut coupling has
  Littlewood--Richardson multiplicity one. The full orbital irrep still has
  `binom(D,N)` components.
- The counterexample rank is invariant under invertible orbital changes and
  internal FEMPS gauges.
- It remains unchanged under direct embedding in a larger orbital space.
- Its one-cut rank is full, so a nonzero maximal minor gives an open
  neighborhood with the same nondivisibility.

Hamiltonian-specific symmetry adaptation remains useful, but its multiplicity
spaces describe physical charges/multiplets, not a universal exchange carrier.

## State-adaptive and truncation alternatives

A decomposable state has an intrinsic occupied `N`-plane, whose exterior powers
explain its binomial cut support. General states do not have a canonical
occupied `N`-plane. Choosing a Slater/secant expansion instead is established
CI/Grassmannian geometry; identifiability is regime-dependent, and shared
physical inputs impose nonfree channel relations.

The ordinary exterior flattening singular values remain the canonical orbital-
unitary spectrum. They reproduce the flat Slater spectrum and therefore do not
remove the Phase 1 exchange floor. A chosen nonorthogonal Slater/AGP expansion
requires its overlap Gram matrix and an explicit error bound. The existing
contribution-Gram spectrum is invariant under term rescaling/permutation for a
supplied expansion, not under every nonlinear decomposition of the same state.

## Contraction check

The Phase 22 path state is a scalar multiple of one top Slater, so its desired
correlation multiplicity is one. Its scalar and squared norm nevertheless
encode a permanent. A canonical state label cannot by itself recover the state
from compact cores or evaluate the required norm. This independently fails the
algorithmic Gate L criterion.

## Exact certificate

The implementation-independent verifier uses exact rational exterior
flattenings for every `3<=N<=8`. It checks all cuts for the Slater and
counterexample, the component-channel deficit, reversed orbital permutations,
direct orbital embeddings, and a full-support perturbation. Its hash is

```text
06025f168a49c0ab857c2163103ffabcb56fb04cd1fed4df9120d25ef6bc60df
```

## Prior art and novelty

Structural/degeneracy tensor decomposition, non-Abelian QSpace, two-fermion
Slater rank, and Grassmannian-secant identifiability are prior art. The
project-specific contribution is the direct rank-divisibility counterexample,
its compact-FEMPS contraction consequence, and the resulting scope correction.

## Verification

- Exact-certificate tests: `8 passed`.
- Complete repository suite: `205 passed` with one unchanged latticeTN
  report-path warning.
- The four-page theorem draft compiles twice with resolved references and no
  overfull, underfull, or undefined warnings.
- `git diff --check` passes apart from platform line-ending notices.

## Decision

ADR 0015 closes Candidate L1 negatively. Phase 26 will consolidate the full
no-go and alternative-route package into a single internally reviewable
manuscript, with explicit external-proof-review boundaries.
