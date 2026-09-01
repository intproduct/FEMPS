# Statistics-carrier/correlation-multiplicity audit

## Intended analogy

Symmetry-adapted tensor networks decompose an invariant tensor into fixed
group-theoretic structural tensors and free degeneracy tensors. Singh,
Pfeifer, and Vidal state this distinction explicitly
[@SinghPfeiferVidal2010SymmetryTN], and Weichselbaum develops its non-Abelian
Clebsch--Gordan/Wigner--Eckart implementation [@Weichselbaum2012NonAbelianTN].
This is the closest established analogue to the proposed FEMPS statistics
carrier. It does not automatically supply the desired fermionic factorization.

## Why particle antisymmetry is different

Under particle permutations, a fully antisymmetric state transforms in the
one-dimensional sign representation. Its restriction to
`S_k x S_(N-k)` is `sgn_k tensor sgn_(N-k)`, also with multiplicity one. The
`binom(N,k)` particle-Schmidt rank of a Slater determinant is therefore not the
dimension of an `S_N` structural irrep or a symmetry degeneracy space.

Under one-particle basis changes, the state space `Lambda^N V` is an
irreducible `GL(V)` representation. The exterior coproduct embeds it in
`Lambda^k V tensor Lambda^(N-k) V`; the relevant Littlewood--Richardson
multiplicity is one. A symmetry-adapted description can put the orbital
components on an external `Lambda^N V` irrep leg, but that leg has dimension
`binom(D,N)` and contains the full state rather than only exchange bookkeeping.

Thus the two natural symmetries give either a one-dimensional sign carrier or
the full orbital exterior representation. Neither gives a fixed
`binom(N,k)` carrier with a smaller free correlation multiplicity.

## State-adaptive alternatives

For a decomposable state `C=u_1 wedge ... wedge u_N`, its occupied support
`U` is intrinsic and the cut spaces are `Lambda^k U` and
`Lambda^(N-k) U`. This explains the binomial Slater rank, but `U` is a
state-dependent orbital subspace, not a universal exchange tensor.

For a finite Slater sum, carrying one such structural block per determinant is
ordinary CI/secant structure. Shared physical contraction inputs impose
state-dependent relations among the component blocks, so they do not form a
free canonical tensor product. Slater decomposition/rank is established for two fermions
[@SchliemannCiracKusEtAl2001SlaterRank]; at higher degree the relevant
Grassmannian secant identifiability problem has a substantial, nonuniform
geometry [@BallicoBernardiCatalisanoChiantini2013GrassmannSecants;
@GalganoStaffolani2024GrassmannianIdentifiability]. A chosen decomposition
therefore cannot be assumed unique or stable.

The project's gauge-balanced contribution Gram spectrum is invariant to term
rescaling and permutation inside one supplied finite-AGP expansion. It is not
proved invariant under all alternative nonlinear decompositions of the same
state and must not be promoted to a canonical state spectrum.

## Audit conclusion

The direct tensor-product proposal fails an elementary dimension test. More
flexible symmetry-sector direct sums are valid established technology, but
they do not isolate the Slater binomial as an `S_N` degeneracy. State-adaptive
Slater/secant channels reduce to established determinant expansions and inherit
overlap, identifiability, and contraction issues. A future carrier proposal
must specify a different category and show explicitly why its construction
does not hide the ordinary exterior coefficient problem.
