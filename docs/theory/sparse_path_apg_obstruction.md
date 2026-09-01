# Sparse growing-width pair paths are already APG-hard

## Status

This note closes the Phase 22 sparse growing-width candidate negatively. The
state identification and permanent coefficient identity are elementary exact
theorems. Their bounded verifier uses independent exact-integer routes. The
complexity conclusion uses Valiant's standard #P-completeness theorem and is
therefore a conditional obstruction to polynomial exact contraction, not an
unconditional separation of complexity classes.

The APG/APIG state families, their permanent-valued determinant coefficients,
and Fischer/Waring expansions are established prior art. The project-specific
result is their embedding as the weakest fixed-bandwidth matrix-pair candidate
remaining after Phase 21 and the resulting Gate J classification.

## Candidate family and parameters

Let `V_D` have an orthonormal basis and let

```text
Omega in Mat_w(Lambda^2 V_D),
l,r in C^w.
```

For `N=2M`, define the normalized matrix-pair state

```text
Psi_M(Omega;l,r) = l^T Omega^M r / M!.
```

Candidate J allows `w` to grow jointly with `(M,D)`, while the virtual matrix
is tridiagonal or has a fixed bandwidth. For a path graph, its exact formal
walk recurrence is

```text
X_0(b) = delta_(b,b0),
X_(t+1)(b) = sum_(a: |a-b|<=q) X_t(a) wedge Omega_(a,b),
```

where `q` is the fixed half-bandwidth. This is a compact arithmetic branching
program if the two-forms are treated as formal edge labels. It is not yet a
physical amplitude, norm, or Hamiltonian recurrence: `X_t(b)` is a `2t`-form,
whose exact coefficient support can have size `binom(D,2t)`.

The sharp obstruction already occurs in the upper-bidiagonal subclass

```text
w = M+1,
Omega_(i-1,i) = F_i,       1 <= i <= M,
Omega_(a,b) = 0            otherwise,
l = e_0,
r = e_M.
```

Its sparse input has `M binom(D,2)` scalar pair coefficients in general. The
hard specialization below uses only `D=2M` and `M^2` binary coefficients.

The usual similarity gauge acts as

```text
Omega -> G^-1 Omega G,
l^T -> l^T G,
r -> G^-1 r.
```

Diagonal `G` preserves the displayed path sparsity and rescales neighboring
edges. With fixed endpoint boundaries, internal vertex scalings give `M-1`
continuous gauge directions. Because even forms commute, permutation of the
`F_i` and edge rescalings whose product is one are additional state-level
parameter redundancies. No canonical form is inferred from this observation.

## Proposition 1: the unique path is APG

For the upper-bidiagonal family,

```text
(Omega^M)_(0,M) = F_1 wedge ... wedge F_M.
```

**Proof.** A nonzero length-`M` matrix product from vertex `0` to vertex `M`
must advance by one at every multiplication. Thus there is exactly one virtual
path and its edge product is the displayed exterior product. QED.

This is precisely an antisymmetrized product of geminals (APG), apart from the
project's `1/M!` normalization. A general fixed-bandwidth endpoint coefficient
is a sum over walk-indexed APG products. Sparse matrix powers therefore give a
structured APG sum, not a new fermionic state primitive.

Fischer's identity gives the formal exact expansion

```text
F_1 ... F_M
 = 1 / (2^(M-1) M!)
   sum_(epsilon_2,...,epsilon_M in {+1,-1})
   (product_(i=2)^M epsilon_i)
   (F_1 + sum_(i=2)^M epsilon_i F_i)^M.
```

It applies because physical two-forms commute. Kawasaki--Nakatani use this
identity directly for APG-to-AGP-CI conversion
[@KawasakiNakatani2024LowRankAPG]. It is an exponential exact upper bound, not
a minimal-rank theorem in the exterior quotient. In particular, the ordinary
Waring rank of the formal monomial `F_1...F_M` cannot be used as a physical
LC-AGP lower bound: the map

```text
Sym^M(Lambda^2 V_D) -> Lambda^(2M) V_D
```

has a large kernel. For example, products of disjoint decomposable pairs can
collapse to one AGP power.

## Theorem 2: permanent-valued top coefficient

Let `D=2M`, define orthonormal decomposable pair forms

```text
P_j = e_(2j-1) wedge e_(2j),       1 <= j <= M,
```

and let `A` be an `M x M` scalar matrix. Set

```text
F_i = sum_(j=1)^M A_(i,j) P_j.
```

Then

```text
F_1 wedge ... wedge F_M
  = perm(A) P_1 wedge ... wedge P_M,
```

and hence the normalized sparse-path state satisfies

```text
Psi_M = perm(A) P_1 wedge ... wedge P_M / M!,
||Psi_M||^2 = |perm(A)|^2 / (M!)^2.
```

**Proof.** Expand the product by multilinearity. Since `P_j wedge P_j=0`, a
surviving selection must use every pair label exactly once and is indexed by a
permutation `sigma`. Since the `P_j` have even degree, they commute without a
sign. Every surviving term therefore has the same top form and coefficient
`product_i A_(i,sigma(i))`; summing over `sigma` is the permanent. The top form
has norm one under the project convention. QED.

This is the paired-orbital APIG coefficient formula. Richer--Kim--Ayers derive
permanent-valued APG determinant coefficients and identify the single-pairing-
scheme APIG restriction [@RicherKimAyers2025GraphicalGeminals]. Thus the
permanent structure itself is prior art.

## Corollary 3: exact norm is #P-hard

Restrict `A` to 0--1 entries. Valiant proves that computing `perm(A)` for this
input class is #P-complete [@Valiant1979Permanent]. The sparse-path instance is
constructible using

```text
M pairs, D=2M, w=M+1, bandwidth=1, O(M^2) nonzero scalar input data.
```

An exact squared-norm oracle returns the nonnegative rational

```text
s = perm(A)^2 / (M!)^2.
```

Exact multiplication by `(M!)^2` followed by the nonnegative integer square
root recovers `perm(A)` in polynomial time. Therefore generic exact norm
contraction for this upper-bidiagonal family is #P-hard. A polynomial algorithm
jointly in `(M,D,w)` would imply `FP=#P`.

This reduction is simpler than the Phase 13 tagged noncommutative Cayley-
determinant construction. The virtual path is unique and all virtual matrices
are scalar shift edges; the hardness lies in exterior coefficient extraction
from distinct geminals. Since tridiagonal and every broader fixed-bandwidth
class contain the upper-bidiagonal subclass by zero specialization, they inherit
the same generic exact obstruction.

The reduction does **not** prove a large physical LC-AGP rank. At `D=2M`, the
top exterior space is one-dimensional; once the permanent has been computed,
the resulting state is a scalar multiple of one top form. This is deliberately
a contraction-complexity counterexample, not a representational-rank lower
bound.

## Consequences for norm and observables

The walk recurrence evaluates formal virtual propagation but must retain
exterior support or perform hard coefficient extraction. For the unique-path
case it merely builds the compact product `F_1...F_M`; extracting its one top
coefficient is already the permanent. Thus `O(Mw)` walk counting cannot be
reported as a physical amplitude or norm algorithm.

Because Gate J requires norm plus one- and factorized-two-body contractions,
the norm obstruction alone rejects a generic exact solver. No GPU kernel,
automatic-differentiation optimizer, or continuous variational benchmark is
admitted for this family. Structured restrictions such as AGP, strongly
orthogonal geminals, Cauchy-type APIG coefficients, or controlled approximate
selection remain separate established routes; they are not rescued by sparse
virtual width alone [@MoissetFecteauJohnson2022GeminalRDM;
@RicherKimAyers2025GraphicalGeminals].

## Exact certificate

`math/certificates/verify_sparse_path_apg_permanent.py` imports neither PyTorch
nor `femps`. For identity, all-one, and deterministic binary matrices at every
`1<=M<=6`, it compares:

1. explicit upper-bidiagonal virtual-path propagation;
2. square-zero commuting exterior subset propagation; and
3. independent permutation enumeration of the permanent.

It also checks the normalized squared norm exactly over the rationals. The
certificate hash is

```text
dd72c1aaeb0bc2a6b9206992cde9099f2f568b7ff6c8ed8eb7e38d958f78e790
```

The bounded certificate is regression evidence for the implementation and
normalization. The all-`M` proof is the symbolic argument above.

## Gate J classification

Gate J is **FAIL** for generic growing-width path/fixed-bandwidth pair matrices.
The weakest unique-path member is established APG/APIG rather than a new state
family, while its exact norm contains the 0--1 permanent. The two required gate
conditions therefore fail independently:

1. novelty/separation is not established and the selected state is direct APG
   prior art; and
2. exact joint-polynomial contraction is #P-hard under the standard assumption.

Sparse virtual degree, compact parameter count, and cheap formal walk
propagation are not sufficient contraction criteria.
