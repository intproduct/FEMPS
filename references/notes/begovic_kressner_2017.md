# Begovic Kovac--Kressner 2017: structured antisymmetric approximation

## Representation being compressed

The paper studies an order-`N` antisymmetric tensor through its **modewise
multilinear (Tucker) rank**. Antisymmetry forces all single-mode ranks to be the
same. A structure-preserving approximation uses a common one-particle subspace
matrix `U` on every mode and an antisymmetric core:

\[
  B=S\times_1 U\times_2\cdots\times_N U,
  \qquad S\in\Lambda^N\mathbb R^r.
\]

The algorithms therefore seek a smaller one-particle support while keeping the
full tensor alternating. The paper adapts HOOI/Jacobi-style methods and gives
special treatment to the minimal nonzero multilinear rank `r=N`, where the
approximant is decomposable.

## Boundary with the FEMPS no-go statement

This does not contradict the ordinary particle-TT rank floor. Tucker rank `r`
is the rank of a one-mode-versus-all-other-modes unfolding and measures the
dimension of the occupied one-particle support. The no-go theorem concerns a
`k`-particles-versus-`N-k`-particles unfolding. Any nonzero alternating Tucker
core still has ordinary particle-cut rank at least `{N choose k}`.

Thus structure-preserving Tucker approximation can be effective when the
one-particle support is compressible, while it does not remove the binomial
exchange multiplicity paid by a particle-site TT. The two results answer
different rank questions.

## Source status

Checked against the authors' preprint, arXiv:1603.05010, and the published SIAM
abstract/metadata. This note is a representation boundary, not a novelty claim.
