# Active execution plan: Phase 21 growing-memory exterior classification

## Objective

Test the weakest escape from the Phase 20 collapse theorem: a radical
nilpotency depth that grows with the problem while semisimple blocks remain
scalar. Determine whether controlled one-generator path memory produces a
genuinely beyond-LC-AGP exact family, or prove that it too admits a jointly
polynomial LC-AGP expansion before considering minimally noncommutative
growing-memory algebras.

## Candidate I1

Use the truncated-polynomial coefficient algebra

```text
A_d = C[z] / (z^d),
Omega(z) = sum_(j=0)^(d-1) F_j z^j,
```

with arbitrary boundary functional on coefficients of `Omega(z)^M mod z^d`.
The radical `(z)` has nilpotency index `d`, so this family lies outside Phase
20 when `d` grows, while retaining a single commutative memory coordinate.

## Checkpoints

- [x] Derive an exact coefficient-extraction/LC-AGP formula and a term bound
  polynomial jointly in `M,d,D`, or exhibit an obstruction.
- [x] Prove arbitrary-boundary coverage and distinguish exact Waring rank from
  border-rank/derivative limits.
- [x] Build an implementation-independent exact-rational certificate for
  bounded `(M,d)` cases; do not rely on floating-point root-of-unity cancellation.
- [x] Cross-check small exterior states and admitted-parameter gradients only
  if the symbolic identity needs a numerical implementation oracle. Not
  required: the verifier compares the full exterior-even polynomial identity
  for every boundary basis element without a floating-point implementation.
- [x] Audit the result against AGP-CI, binary-form Waring decomposition, jet/
  confluent interpolation, and commutative-algebra moment-state literature.
- [ ] If I1 collapses, formulate the smallest two-generator or quiver/path
  candidate whose memory depth grows but whose graph width remains controlled.
- [ ] Issue Gate I without starting continuous variational experiments unless
  a family is both beyond polynomial-size LC-AGP and exactly contractible.

## Exit criterion

A growing nilpotency index is only a necessary escape condition from Phase 20,
not evidence of novelty. Gate I passes only with a joint polynomial contraction
whose state family cannot be reorganized as polynomial-size finite LC-AGP or a
standard Gaussian/Pfaffian construction. A further negative classification is
an acceptable result and should narrow the next algebra rather than trigger an
uncontrolled numerical search.
