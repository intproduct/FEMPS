# Statistics-carrier/correlation-multiplicity obstruction

## Status and object being tested

Phase 25 tests the literal master-plan factorization at a particle cut. Let
`0 != C in Lambda^N V` and define the exterior contraction/flattening map

```text
c_(C,k): Lambda^k V* -> Lambda^(N-k) V,
c_(C,k)(alpha) = i_alpha C.
```

Its rank `r_k(C)` is the ordinary particle-cut Schmidt rank, up to the fixed
normalization of the exterior coproduct. Candidate L1 asks for an exact
factorization of its Schmidt support

```text
B_k(C) ~= S_(N,k)^fermion tensor M_k(C),                 (1)
```

where the structural carrier is fixed by `(N,k)`, a Slater has
`dim M_k=1`, and `M_k` is canonical up to unitary equivalence. Reconstruction,
contraction, and safe truncation are additional requirements.

This note rejects (1) for all states and then audits the natural weaker
interpretations. It does not reject every possible categorical or
Hamiltonian-specific use of symmetry.

## Proposition 1: a direct tensor product fails by dimension

For every `N>=3`, no factorization (1) can be exact for all nonzero
`N`-forms if its structural carrier reproduces the Slater particle-cut support
and its multiplicity is a finite-dimensional vector space.

### Proof

At the `1|N-1` cut, every nonzero Slater determinant has

```text
r_1 = N.
```

The condition `dim M_1=1` therefore forces `dim S_(N,1)^fermion=N`. Every
state covered by (1) would consequently have `r_1` divisible by `N`.

Take `V` of dimension `N+2`, put

```text
W = e_6 wedge ... wedge e_(N+2)
```

with `W=1` when `N=3`, and define

```text
C_N = e_1 wedge (e_2 wedge e_3 + e_4 wedge e_5) wedge W.   (2)
```

The `N+2` one-index contractions of (2) are linearly independent. The
contractions by `e_2*,...,e_5*` contain four distinct monomials with all of
`W`; contraction by `e_1*` is the only one without `e_1`; and contractions by
the covectors dual to factors of `W` have distinct missing `W` factors. Hence

```text
r_1(C_N)=N+2.
```

Since `N` does not divide `N+2` for `N>=3`, (1) is impossible. QED.

The obstruction is not a coordinate or isolated-rank artifact. Exterior
flattening rank is invariant under every invertible orbital transformation and
under internal FEMPS gauges because both act by invertible changes of basis or
leave `C` unchanged. Embedding (2) into a larger orbital space keeps its
intrinsic support and rank. Moreover `r_1=N+2=dim V` is full row rank, so a
nonzero maximal minor persists on a Euclidean- and Zariski-open neighborhood
of (2). A generic small perturbation does not restore divisibility.

The form (2) is only a sum of two Slaters. Each component has rank `N`, and in
this coordinate example their contraction-image spaces have a `2N`-dimensional
direct sum if their input covectors are varied independently. In the physical
sum the same covector acts on both components, locking the `N-2` shared-orbital
channels; the image has rank `N+2`, not `2N`. These state-dependent relations
are exactly what a fixed carrier tensored with a free two-dimensional
multiplicity would miss.

## Representation-theory audit

The standard structural/degeneracy split of a symmetric tensor is real and
useful [@SinghPfeiferVidal2010SymmetryTN;
@Weichselbaum2012NonAbelianTN]. Its application here gives two different
answers, neither Candidate L1:

1. Under particle permutations, the fermionic sector is the one-dimensional
   sign representation. Its restriction to `S_k x S_(N-k)` is
   `sgn_k tensor sgn_(N-k)` with multiplicity one. The Slater factor
   `binom(N,k)` is not an `S_N` irrep dimension or degeneracy.
2. Under `GL(V)`, `Lambda^N V` is irreducible. The exterior coproduct into
   `Lambda^k V tensor Lambda^(N-k) V` is the unique equivariant coupling up to
   scale because the corresponding Littlewood--Richardson multiplicity is one.
   Putting the state on its covariant external irrep leg retains
   `binom(D,N)` orbital components; it has not separated exchange from
   correlation.

Using an actual orbital symmetry subgroup of a Hamiltonian may create useful
charge/multiplet degeneracy spaces, exactly as in symmetry-adapted TN. Those
spaces depend on the physical symmetry and sector content. They are not a
universal fermionic exchange carrier and need not give multiplicity one for a
Slater.

## State-adaptive routes and why they are not (1)

For a decomposable `C`, the minimal occupied support `U(C)` is intrinsic and

```text
im c_(C,k) = Lambda^(N-k) U(C).
```

This supplies a state-dependent binomial structural space and the desired
Slater sanity check. For a correlated form there is no canonical occupied
`N`-plane. Replacing it by the full support makes the alleged carrier large
and gives multiplicity one to every state without measuring correlation.

Alternatively, choose a Slater/secant expansion. Slater rank and the special
two-fermion canonical decomposition are established notions
[@SchliemannCiracKusEtAl2001SlaterRank]. In higher degree, uniqueness is a
Grassmannian-secant identifiability question and holds only in classified
regimes, not by default [@BallicoBernardiCatalisanoChiantini2013GrassmannSecants;
@GalganoStaffolani2024GrassmannianIdentifiability]. Even when a decomposition
is supplied, overlapping cut spaces give a sum/quotient rather than a tensor
product, as (2) shows. Computationally this is a finite Slater/AGP expansion,
which is already a project baseline and direct prior art.

## Canonical truncation does not emerge

The singular values of `c_(C,k)` are canonical under orbital unitaries and
give the usual particle Schmidt error bound. They are therefore the honest
state-intrinsic truncation spectrum, but for a Slater they contain
`binom(N,k)` equal nonzero values. Treating those equal values as an implicit
carrier works only on the decomposable orbit; a generic perturbation splits
the degeneracy and supplies no canonical grouping into carrier copies.

A chosen nonorthogonal Slater/AGP expansion can be truncated only after its
overlap Gram matrix and the resulting state/energy error are controlled. The
project's contribution-Gram diagnostic is invariant under term rescaling and
permutation for one supplied expansion, not under every alternative nonlinear
decomposition. It is not a gauge-independent spectrum of `C`.

Thus no safe multiplicity truncation theorem follows from symmetry. Exterior-
coefficient or ordinary Schmidt truncation can preserve antisymmetry when
performed within `Lambda^N V`, but it does not remove the Slater flat-spectrum
cost proved in Phase 1.

## Contraction obstruction survives a perfect Slater label

The Phase 22 sparse path provides an independent algorithmic check. With
`D=N=2M`, its output is

```text
Psi_A = perm(A) e_1 wedge ... wedge e_N / M!.
```

Whenever nonzero, this is projectively one Slater, so any state-intrinsic
correlation multiplicity satisfying the sanity condition is one at every cut.
Nevertheless, obtaining its scale and squared norm from the compact path input
computes the permanent exactly, and generic relative norm approximation also
contains the real-PSD permanent obstruction. A state-level label
`multiplicity=1` therefore does not compile compact FEMPS cores into a
polynomial contraction. The hard scalar remains outside the proposed carrier.

This observation does not claim that every normalized observable on this
one-dimensional example is hard; it proves that Candidate L1's required norm
and reconstruction algorithm cannot follow from the intrinsic multiplicity
alone.

## Gate L decision

**Gate L: FAIL for the direct generic statistics-carrier tensor product.**

It fails three independent exit conditions:

1. exact factorization: cut ranks need not be divisible by the Slater carrier
   dimension;
2. canonicality: permutation symmetry has only a one-dimensional sign irrep,
   while orbital symmetry leaves the full exterior irrep or depends on an
   external Hamiltonian symmetry; and
3. contraction: even a projective single-Slater output can hide a permanent in
   the scalar produced from compact matrix-pair cores.

Symmetry-adapted occupation/charge TNs, finite Slater/AGP sums, and
Hamiltonian-specific orbital symmetries remain valid established methods. They
must not be relabeled as the missing universal FEMPS correlation-multiplicity
spectrum.
