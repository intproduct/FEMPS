# Ordinary particle-TT no-go theorem package

## Scope and convention

Let `V` be a finite-dimensional real or complex Hilbert space. We identify an
alternating `N`-particle coefficient tensor with an element

\[
    C\in\Lambda^N V\subset V^{\otimes N}.
\]

For orthonormal orbitals `u_1,...,u_N`, the normalized Slater convention is

\[
 \Phi=\frac{1}{\sqrt{N!}}\sum_{\pi\in S_N}\operatorname{sgn}(\pi)
 u_{\pi(1)}\otimes\cdots\otimes u_{\pi(N)}.
\]

Thus `||Phi||=1`. This document concerns an **ordinary particle-site TT**:
tensor axis `j` stores the coordinate/basis index of particle `j`. It does not
rule out ordered-sector, determinant-sum, Fock-space, or exterior-carrier
representations.

## Theorem 1: exact TT rank is particle-unfolding rank

For `1 <= k < N`, let `C_(k)` be the matricization

\[
 C_{(k)}:V^{\otimes k}\longrightarrow V^{\otimes(N-k)}
\]

obtained by grouping the first `k` and last `N-k` indices. The minimal exact
ordinary-TT bond at this cut is

\[
 r_k^{\mathrm{TT}}(C)=\operatorname{rank} C_{(k)}.
\]

**Proof.** Cutting any TT at bond `k` expresses `C_(k)` as a product of a left
and a right interface matrix, so its matrix rank cannot exceed the bond size.
Conversely, the sequential exact SVD construction produces one TT whose bond
sizes are precisely the ranks of all successive unfoldings. This is the
standard TT rank characterization.

## Theorem 2: universal exchange rank floor

If `0 != C in Lambda^N V`, then every particle cut obeys

\[
 \operatorname{rank} C_{(k)}\geq {N\choose k}.
\]

**Proof.** Choose a basis coefficient `c_I != 0`, where `I` is an `N`-element
set. Index rows by the `k`-subsets `A` of `I`, and index the corresponding
columns by `I\A`. In the resulting square submatrix, the entry pairing `A`
with `I\A` is `+/- c_I`. If `A != A'`, the concatenation of `A` with `I\A'`
repeats at least one index, so its alternating coefficient is zero. The
submatrix is diagonal up to row/column signs and has size `{N choose k}`.

The bound is sharp: a nonzero decomposable wedge supported on an
`N`-dimensional subspace has rank exactly `{N choose k}`.

### Corollary: strict symmetry cannot survive a lower bond

If a nonzero ordinary TT tensor `C_tilde` has

\[
 r_k^{\mathrm{TT}}(\widetilde C)<{N\choose k}
\]

at any cut, then `C_tilde` is not fully alternating. Therefore a plain TT-SVD
truncation below this floor necessarily breaks the full `S_N` sign
representation, even if some within-block antisymmetry happens to remain.

## Theorem 3: flat particle-Schmidt spectrum of a Slater state

For every `k`-subset `I` of `{1,...,N}`, let `Phi_I` and `Phi_{I^c}` be the
normalized Slater states made from those orbital subsets. Then

\[
 \Phi=\frac{1}{\sqrt{{N\choose k}}}
 \sum_{|I|=k}(-1)^{\epsilon(I)}\Phi_I\otimes\Phi_{I^c},
\]

where the shuffle sign moves the orbitals in `I` before those in `I^c`.
The left states are mutually orthonormal, as are the right states. Hence the
nonzero singular values of `Phi_(k)` are

\[
 \lambda_I={N\choose k}^{-1/2},\qquad |I|=k.
\]

**Proof.** Partition the signed permutation sum by the set of orbitals placed
in the first `k` slots. Each class factors into the normalized internal
antisymmetrizers `Phi_I` and `Phi_{I^c}`. The coefficient is
`sqrt(k!(N-k)!/N!)=1/sqrt({N choose k})`. Orthogonality follows from the
orthonormal orbital assumption.

## Corollary: approximate ordinary-TT lower bound

Let `m={N choose k}`. By Eckart--Young, the best tensor constrained only to
matrix rank at most `r` across this particle cut has relative squared error

\[
 \epsilon_{k,r}^2=\max\left(0,1-\frac{r}{m}\right).
\]

Consequently every ordinary TT whose `k`th bond is at most `r` satisfies the
same expression as a lower bound. To achieve relative error at most `epsilon`
at that cut, it is necessary that

\[
 r\geq\left\lceil(1-\epsilon^2){N\choose k}\right\rceil.
\]

This is an exchange-statistics cost already present in a single uncorrelated
Slater determinant. It is not a claim that every correlated alternating
tensor has a flat spectrum; general alternating tensors may have additional
rank and nonuniform singular values.

## What this does and does not say about structured approximation

Structure-preserving multilinear/Tucker approximation uses the same
one-particle subspace matrix on every tensor mode and retains an alternating
core. Its rank parameter measures one-mode support dimension; it is not the
rank of a `k`-particle unfolding. A nonzero alternating Tucker core can reduce
one-particle support but still obeys the binomial particle-cut floor. A sum of
Slater determinants, an ordered-coordinate sector, and a Fock-space occupation
MPS likewise reorganize the degrees of freedom. None is equivalent to lowering
an ordinary particle-cut TT bond below the floor while retaining a nonzero
full-space alternating tensor.

The precise comparison target is Begovic Kovac--Kressner (2017), while the TT
rank identity uses the standard characterization of Oseledets (2011). The
FEMPS motivation is therefore representation-specific, not a universal
impossibility theorem for fermionic tensor networks.

## Executable evidence

`femps.exterior.reference` implements two independent Slater materializations:
direct antisymmetrization and determinant minors. Tests verify their agreement,
unit norm, all-transposition antisymmetry through adjacent generators, the
flat spectra at every cut for `N=4`, the binomial rank floor, the closed-form
truncation error, and symmetry loss after an ordinary sub-floor SVD truncation.

The implementation is exponential by design and is restricted to small-system
truth-oracle use.

## Open extensions

- Stronger contraction-rank bounds under concise/full-support hypotheses.
- Achievable rank spectra for general alternating forms.
- Approximate-rank statistics beyond decomposable Slater tensors.
- Sharper lower bounds tied to genuine fermionic correlation rather than
  exchange alone.
