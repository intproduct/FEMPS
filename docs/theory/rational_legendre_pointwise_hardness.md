# Exact pointwise hardness in the rational Legendre basis

## Scientific status

The statement below remains a **conjecture** with an internally complete proof
draft. The submission manuscript continues to use that evidence label until an
external algebraic-complexity researcher has checked
the encoding and reduction. It is not a statement about approximate sampling,
VMC, QMC, or all `chi>=2` FEMPS.

## Problem convention

Let `P_r(x)` be the standard unnormalized Legendre polynomial, with
`P_r(1)=1`; no orthonormal factor `sqrt((2r+1)/2)` is introduced. Inputs are
rational `2 x 2` Legendre coefficient matrices for site-labelled one-particle
functions

```text
F_i(x) = sum_{r=0}^{D-1} B_[i,r] P_r(x),
```

together with rational boundaries `u,v` and rational points
`x_1,...,x_N in (-1,1)`. The requested value is the unnormalized alternating
point value

```text
Alt(F_1,...,F_N)(x_1,...,x_N)
 = sum_{sigma in S_N} sgn(sigma)
   u^T F_1(x_sigma(1)) ... F_N(x_sigma(N)) v.
```

This is the coordinate evaluation naturally associated with the algebraic
exterior product. If a physics convention inserts a known global
`1/sqrt(N!)`, the theorem should be stated for the unnormalized value above;
doing so keeps the entire Turing model rational and uses no algebraic-number
oracle.

## Rational interpolation lemma

For `D=n`, choose

```text
xi_j = -1 + 2j/(n+1),  j=1,...,n.
```

These are distinct rationals in `(-1,1)` with `O(log n)`-bit numerators and
denominators. Let

```text
E_[j,r+1] = P_r(xi_j),  0<=r<n.
```

The matrix `E` is nonsingular: `P_0,...,P_(n-1)` are a degree-graded basis of
the polynomials of degree below `n`, and a nonzero polynomial of that degree
cannot vanish at all `n` distinct nodes.

The explicit formula

```text
P_r(x) = 2^(-r) sum_{q=0}^{floor(r/2)}
         (-1)^q binom(r,q) binom(2r-2q,r) x^(r-2q)
```

shows that every entry of `E` has polynomial bit length (in fact
`O(n log n)` is sufficient). Choose a common denominator `Q` for all entries;
even the product of their individual denominators has polynomial bit length.
Then `B=Q E` is an integer matrix whose entries have polynomial bit length.
Hadamard's bound gives polynomial bit length for `det(B)` and every
`(n-1) x (n-1)` minor. Cramer's rule,

```text
E^(-1) = Q adj(B)/det(B),
```

therefore gives rational entries of polynomial bit length and is computable by
exact rational Gaussian elimination in polynomial time. This supplies the
required inverse-matrix bound, rather than merely invoking interpolation over
an unspecified field.

Define the Lagrange functions in the requested basis by

```text
L_j(x) = sum_{r=0}^{n-1} (E^(-1))_[r+1,j] P_r(x).
```

Their Legendre coefficients have polynomial bit length and
`L_j(xi_k)=delta_jk` exactly.

## Reduction from CHSS

Given a 3CNF formula `phi`, apply the polynomial-time CHSS construction to
obtain an `n x n` matrix `H=(H_ij)` over `Mat_2(Q)` such that

```text
CDet(H)=a I_2+b J_2,  a+b=4^(3m) #SAT(phi).
```

Set

```text
F_i(x) = sum_{j=1}^n H_ij L_j(x),
u=e_1,  v=e_1+e_2.
```

Exact rational matrix arithmetic constructs all Legendre coefficients in
polynomial time and polynomial bit length. At the ordered tuple
`(xi_1,...,xi_n)`, the Kronecker interpolation identities give

```text
Alt(F_1,...,F_n)(xi_1,...,xi_n)
 = sum_sigma sgn(sigma) u^T H_[1,sigma(1)] ... H_[n,sigma(n)] v
 = u^T CDet(H) v
 = 4^(3m) #SAT(phi).
```

Thus one exact point-evaluation query followed by exact division by `4^(3m)`
recovers `#SAT(phi)`. The CHSS matrix dimension is polynomial in `|phi|`, the
nodes, Legendre coefficients, query, and answer all have polynomial bit length,
and every virtual matrix is `2 x 2`. The construction is a polynomial-time
metric reduction, hence also a polynomial-time Turing reduction.

## Conjecture under external review

> Exact unnormalized pointwise evaluation of rational first-quantized
> continuous one-form FEMPS represented in the standard unnormalized Legendre
> basis is `#P`-hard under polynomial-time metric reductions, already at
> maximum internal bond two and `D=N`.

The statement must remain marked conjectural in submission-facing text
until external review confirms (i) the CHSS encoding specialization to `Q`,
(ii) the normalization convention, and (iii) the inverse-matrix bit bound.
