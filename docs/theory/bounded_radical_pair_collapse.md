# Bounded-radical matrix-pair candidates: the T2 collapse

## Status

The `2 x 2` upper-triangular coefficient-algebra result below is a proved
algebraic identity with an implementation-independent exact-rational
certificate. It classifies the first Phase 20 candidate as tractable but not
beyond polynomial-size LC-AGP organization. The proposed generalization to
arbitrary bounded radical nilpotency remains a conjecture.

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

## Scope and next conjecture

No claim is made here for all upper-triangular sizes or all algebras with
commutative semisimple quotient. The next candidate theorem is:

> If a split finite-dimensional coefficient algebra has commutative
> semisimple quotient and radical nilpotency index bounded by a constant
> independent of `(N,D,chi)`, then its matrix-pair states have a polynomial-size
> exact LC-AGP expansion.

The intended proof expands by the number of radical insertions (strictly below
the nilpotency index), resolves semisimple idempotent paths, and polarizes the
resulting bounded-variable homogeneous two-form monomials. Until the term
count is derived jointly in algebra dimension and M, this remains a conjecture.
