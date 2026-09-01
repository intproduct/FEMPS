# Fixed-state graded memory collapses to polynomial LC-AGP

## Status

The theorem draft below generalizes the Phase 21 one-generator result to a
fixed matrix-state width and a fixed number of commuting grading counters. It
classifies the first genuinely noncommutative growing-radical candidate
negatively. Its smallest `w=2,g=1` alternating-word instance has an independent
exact-rational certificate. External algebraic review is still required.

Finite weighted automata, matrix linear representations, and their equivalence
with recognizable/rational noncommutative series are established theory
[@BallePanangadenPrecup2015WeightedAutomata;
@BellSmertnig2021NoncommutativePolya]. The project claim is only the LC-AGP
collapse obtained after combining such a bounded-state representation with
physical even-form commutativity and exact coefficient interpolation.

## Fixed-state graded collapse theorem draft

**Theorem draft.** Suppose a family of finite-dimensional complex coefficient
algebras has a faithful graded representation

```text
rho: A -> Mat_w(C[z_1,...,z_g]/(z_1^(d_1),...,z_g^(d_g))),
```

where the entries of `rho(Omega)` have degree at most `d_j-1` in counter
`z_j`. Let `w` and `g` be
constants independent of `(M,D,d_1,...,d_g)`. For any boundary functional
`lambda:A->C`, the pair state `lambda(Omega^M)/M!` is exact LC-AGP with

```text
K <= product_(j=1)^g [M(d_j-1)+1]
     * binom(M+w^2-1,w^2-1).
```

It is therefore polynomial jointly in `M,D` and the explicit graded
representation size `w product_j d_j`. A standard shift-matrix realization has
virtual dimension proportional to that size.

**Proof.** Extend `lambda` from the embedded algebra to a linear functional on
the finite coefficient matrices. Before quotient truncation, every entry of
`rho(Omega)^M` has degree at most `M(d_j-1)` in `z_j`. A tensor-product
Vandermonde grid with `L_j=M(d_j-1)+1` distinct nodes in counter `j` reconstructs
every required coefficient exactly and annihilates every coefficient discarded
by the quotient. Combining the extended boundary with the interpolation weights
leaves at most `product_j L_j` scalar functionals of evaluated `w x w` matrix
powers.

At one grid point, applying any matrix boundary to the M-th power gives a
homogeneous degree-M polynomial in the `w^2` scalar physical two-form entries.
M-th powers of linear forms span this polynomial space, so at most

```text
dim Sym^M(C^(w^2)) = binom(M+w^2-1,w^2-1)
```

scalar AGPs are needed. Multiplying the grid and matrix-power bounds proves the
claim. QED.

The construction is exact in characteristic zero and can use rational nodes
for rational input. As in Phase 21 I1, the arithmetic identity does not imply
that naive Vandermonde coordinates are numerically well conditioned.

## Minimal noncommutative instance

Let

```text
B_d = C<x,y> / (x^2, y^2, every word of length d).
```

Its basis is the identity plus the two alternating words at every length
`1,...,d-1`, hence `dim B_d=2d-1`. The radical has nilpotency index `d`, and
`xy!=yx` for `d>=3`. Define, for a nonempty alternating word `u`,

```text
rho(u) = z^len(u) E_(first(u), complement(last(u))),
rho(1) = I_2.
```

Matrix-unit multiplication is nonzero exactly when the last letter of the
first word differs from the first letter of the second word. Thus `rho` is an
injective algebra homomorphism

```text
B_d -> Mat_2(C[z]/(z^d)).
```

The general theorem with `w=2,g=1` gives

```text
K <= [M(d-1)+1] binom(M+3,3) = O(M^4 d).
```

Since `dim B_d=2d-1`, this is jointly polynomial in particle number, basis
dimension, and coefficient-algebra/virtual representation size. The algebra is
noncommutative and has growing radical depth, yet its two-state alternating
memory remains a polynomial LC-AGP reorganization.

## Exact certificate

`math/certificates/verify_alternating_word_pair_collapse.py` independently:

1. raises a generic symbolic element directly in `B_d`;
2. constructs exact rational z-coefficient weights;
3. decomposes every selected `Mat_2` matrix-power entry by exact four-variable
   simplex interpolation;
4. substitutes the embedded pair forms and compares the complete homogeneous
   polynomial coefficient table; and
5. repeats this for every one of the `2d-1` boundary words at all 12 cases
   `1<=M<=3`, `1<=d<=4`.

The certificate hash is
`082238ff6e6783b7533b3b2a59f3664beb1820794bcae573add98baf43370030`.
It imports neither PyTorch nor `femps`.

## Classification consequence

Growing nilpotency, noncommutativity, two path branches, and constant automaton
width are jointly insufficient. Candidate I2 is rejected as a distinct FEMPS
solver family because its exact contraction is inherited from an explicit
polynomial LC-AGP expansion.

Within this graded-representation route, escaping the present theorem requires
the matrix-state width `w` or the number of independent grading counters `g` to
grow, or requires a representation not reducible to fixed-width matrices over a
fixed-dimensional commutative grading algebra. This is a necessary boundary,
not evidence that every growing-width or growing-counter family is tractable;
Phase 13 already places an unrestricted growing-memory construction on the hard
side.
