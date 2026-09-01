# Bounded Wedderburn--radical matrix-pair collapse

## Status

The `2 x 2` upper-triangular coefficient-algebra result below is a proved
algebraic identity with an implementation-independent exact-rational
certificate. It classifies the first Phase 20 candidate as tractable but not
beyond polynomial-size LC-AGP organization. The subsequent theorem draft
extends the collapse to every finite-dimensional complex coefficient algebra
whose largest semisimple matrix block and radical nilpotency index are both
uniformly bounded. The theorem draft awaits external algebraic review; its
`Mat_2` and `T_2` base cases have independent exact-rational certificates.

## Motivation from noncommutative determinant complexity

For the fields covered by their results, Chien--Harsha--Sinclair--Srinivasan
show that a row-ordered determinant over a finite-dimensional algebra is
efficiently computable when its semisimple quotient is commutative and the
radical has bounded nilpotency index `d`, with runtime `N^O(d)`
[@ChienHarshaSinclairSrinivasan2011NoncommDet]. Their main algebra
classification is stated over finite fields, with a bounded-nilpotency rational
extension in the appendix. For upper-triangular `d x d` matrices, the strictly
upper-triangular radical has index `d`. Thus that theorem is jointly polynomial
only when the nilpotency index is bounded independently of the input; it is not
a polynomial-in-`d` guarantee. The real/complex `T2` result below instead
follows from its own characteristic-zero polynomial identity.

The matrix-pair state uses a fully symmetrized exterior power rather than a
row-ordered determinant. Phase 13's shift tags recover row order by introducing
a nilpotent path memory whose length grows with particle number. Phase 20 asks
whether the determinant-easy, bounded-radical side produces a distinct
contractible fermionic family.

## The smallest noncommuting easy algebra

Let the scalar two-forms `F,G,H` define

```text
Omega = [[F, G],
         [0, H]].
```

These coefficient matrices are generically noncommuting even though the
strictly upper-triangular radical squares to zero. Direct matrix multiplication
over the commutative algebra of physical even forms gives

```text
(Omega^M)[0,0] = F^M,
(Omega^M)[1,1] = H^M,
(Omega^M)[0,1] = sum_(t=0)^(M-1) F^t G H^(M-1-t),
(Omega^M)[1,0] = 0.
```

The normalized matrix-pair state divides every entry by `M!`.

## T2 LC-AGP collapse theorem

**Theorem.** Let every physical coefficient `B_ij` belong to the `2 x 2`
upper-triangular algebra `T2`, and let `l,r` be arbitrary two-component
boundaries. For `N=2M`, the matrix-pair state

```text
l^T Omega_B^M r / M!
```

is an exact LC-AGP with at most

```text
binom(M+2,2) + 2
```

terms. The bound is independent of the one-particle basis dimension `D`.

**Proof.** The two diagonal boundary contributions are scalar AGPs `F^M/M!`
and `H^M/M!`. The only remaining polynomial is

```text
S_M(F,G,H) = sum_(t=0)^(M-1) F^t G H^(M-1-t).
```

Consider the triangular integer grid

```text
Q_M = {(b,c): b>=0, c>=0, b+c<=M}.
```

It has `binom(M+2,2)` points. Evaluation on `Q_M` is unisolvent for bivariate
polynomials of total degree at most M (equivalently, multivariate Newton
interpolation on a lower set). After the nonzero multinomial rescaling, the
coefficient vectors of

```text
(F + b G + c H)^M,  (b,c) in Q_M,
```

therefore form a basis for all degree-M homogeneous polynomials in `F,G,H`.
In particular, exact rational weights `w_bc` exist such that

```text
S_M(F,G,H) = sum_((b,c) in Q_M) w_bc (F+bG+cH)^M.
```

Each term on the right is one scalar fixed-number AGP. Adding the two diagonal
terms proves the bound. The executable construction solves this interpolation
system over exact `Fraction` arithmetic before converting its constants to the
requested tensor dtype. QED.

## Consequences

This result strengthens the earlier observation that simultaneous
triangularizability can create derivative-like mixed terms: for `T2`, those
terms do not escape polynomial-size LC-AGP. With
`K <= binom(M+2,2)+2 = O(M^2)`, the existing exact finite-AGP transition
machinery contracts the norm and one-/factorized-two-body operators in
polynomial cost. This tractability is inherited from an explicit LC-AGP
organization and therefore fails the Phase 20 novelty criterion.

The identity and its reverse-mode derivatives agree only on the admitted
physical-skew, virtual-upper-triangular parameter submanifold. Ambient
derivatives in the forbidden lower virtual block need not agree and are not
physical parameters of this candidate.

## Exact and numerical evidence

`math/certificates/verify_triangular_pair_collapse.py` imports neither PyTorch
nor `femps`. For M=1 through 6 it:

1. raises the symbolic `T2` matrix over `Q[F,G,H]` exactly;
2. solves the triangular-grid interpolation over rational numbers;
3. compares both results coefficient by coefficient; and
4. verifies the committed certificate hash
   `f671c2c10376c39cfb8c223edafba370570b9c417e8b12bcaaeb2f0f66cf078c`.

Independent complex128 tests compare the matrix-pair exterior recurrence to
the constructed LC-AGP sum for M=1,2,3, include explicitly noncommuting
coefficient samples, and compare restricted reverse-mode gradients.

## Full Mat2 semisimple base case

The collapse does not require commutativity of the coefficient algebra. Let
`Omega` be a generic `2 x 2` matrix whose four entries are scalar physical
two-forms. After applying any fixed boundary functional, `Omega^M` is a
homogeneous degree-M polynomial in those four commuting even forms. Over
characteristic zero, M-th powers of linear forms span the space of homogeneous
degree-M polynomials in four variables. Therefore

```text
K <= dim Sym^M(C^4) = binom(M+3,3) = O(M^3)
```

scalar AGPs suffice for every boundary contraction. This is true even though
`Mat_2(C)` is simple and noncommutative. The exact verifier
`math/certificates/verify_mat2_pair_collapse.py` raises the fully generic
symbolic matrix over `Q[x0,x1,x2,x3]`, applies deterministic nonzero rational
boundaries, and reconstructs the result on an integer simplex grid. Its M=1--4
certificate has hash
`74d2de4a2cedcbaf548cd4c9895d0ea4af48e6bb485b72966562e46277a6d20d`.
The certificate is an exact base-case check, while the arbitrary-boundary
statement follows from the spanning argument rather than from that one boundary
sample.

## Bounded Wedderburn--radical collapse theorem draft

**Theorem draft.** Let `A` be a finite-dimensional unital complex algebra with
Jacobson radical `R`, `R^d=0`, and Wedderburn--Malcev decomposition

```text
A = S direct_sum R,   S = direct_sum_(a=1)^q Mat_(p_a)(C).
```

Set `p=max_a p_a` and `rho=dim R`. For `Omega` in
`A tensor Lambda^2(V)`, any linear boundary functional `lambda:A -> C`, and
`N=2M`, the normalized state `lambda(Omega^M)/M!` is an exact LC-AGP with

```text
K <= sum_(k=0)^min(d-1,M)
       q^(k+1) rho^k binom(M+v_k-1,v_k-1),
v_k = (k+1) p^2 + k.
```

When `R=0`, only the `k=0` summand is present. Hence, for fixed `p` and `d`,
the term count and the existing finite-LC-AGP contractions are polynomial
jointly in `M`, `D`, and `dim A` (and therefore in an ambient virtual dimension
`chi`, since `dim A <= chi^2`).

**Proof.** Write `Omega=Omega_S+Omega_R` and expand its M-fold product into
noncommutative words. A word containing `k` radical factors lies in `R^k`, even
when semisimple factors occur between them, so every word with `k>=d` vanishes.
For a surviving `k`, resolve the `k+1` semisimple runs into a path through the
`q` simple blocks and resolve each radical insertion in a basis of `R`. There
are at most `q^(k+1) rho^k` such structural choices.

For any fixed choice, collect the sum over all possible lengths of the
semisimple runs before applying `lambda`. Each run uses at most `p^2` scalar
two-form coordinates and each selected radical insertion contributes one more
scalar two-form. The result is therefore a scalar homogeneous degree-M
polynomial in at most

```text
v_k = (k+1) p^2 + k
```

commuting variables; physical two-forms commute because they have even exterior
degree. In characteristic zero, the Veronese powers `L^M` span
`Sym^M(C^(v_k))`. Selecting a basis of such powers expands the structural
polynomial using at most `binom(M+v_k-1,v_k-1)` terms. After substituting the
actual two-forms back for the formal coordinates, every `L^M/M!` is one scalar
fixed-number AGP. Summing the structural choices gives the stated bound. QED.

This proof is constructive in arithmetic-operation complexity: truncated
polynomial matrix multiplication plus multivariate interpolation produces the
coefficients. It does not assert favorable floating-point conditioning for
every interpolation grid.

## Classification consequence

The theorem is stronger than the original commutative-semisimple-quotient
conjecture: bounded noncommutative simple blocks also collapse. Consequently,
within finite-dimensional matrix-pair coefficient algebras, a family can avoid
this polynomial LC-AGP classification only if its largest simple block size
`p` or radical nilpotency index `d` grows with the problem, or if it leaves the
exact finite-algebra setting. This is a necessary escape condition, not a claim
that every growing-`p` or growing-`d` family is hard or useful.

Phase 13's shift-tag construction uses an upper-triangular path algebra whose
nilpotency index grows with the determinant order, exactly outside the bounded-
`d` theorem. Conversely, the fully noncommutative `Mat_2` pair power remains
easy here despite the permanent-hard row-ordered determinant over `Mat_2`:
symmetrized even-form powers erase the row-order resource unless growing memory
is introduced. Thus neither coefficient noncommutativity nor bounded radical
memory alone supplies the desired beyond-LC-AGP FEMPS family.
