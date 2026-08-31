# Fixed-number Pfaffian FEMPS subclass

## Definition

Let `V_D` be the one-particle functional-basis space and let
`F in C^{D x D}` satisfy `F^T=-F`. Define the pair two-form

\[
 \Omega_F=\frac12\sum_{s,t=1}^D F_{st}e_s\wedge e_t
 =\sum_{s<t}F_{st}e_s\wedge e_t.
\]

For an even particle number `N=2M`, the fixed-number antisymmetrized geminal
power (AGP) state is

\[
 \Psi_M(F)=\frac1{M!}\Omega_F^{\wedge M}\in\Lambda^{2M}V_D.
\]

In the increasing exterior basis its coefficient on a `2M`-subset `I` is

\[
 [\Psi_M(F)]_I=\operatorname{pf}(F_{I,I}).
\]

This is a first-quantized continuous state after substituting
`e_s -> phi_s(x)`. It does not introduce occupation sites.

## Structured matrix-wedge FEMPS embedding

Suppose

\[
 F=\sum_{a=1}^r w_a(u_av_a^T-v_au_a^T),\qquad
 \Omega_F=\sum_{a=1}^r w_a u_a\wedge v_a.
\]

A matrix-wedge chain with internal bond `r` represents `Psi_M(F)` exactly:

1. an odd core selects a pair channel;
2. the following even core emits the matching `v_a` and preserves the label;
3. later odd cores allow only a strictly larger channel label.

Every nonzero virtual path therefore chooses
`a_1<...<a_M`. Its path sum is

\[
 \sum_{a_1<\cdots<a_M}\prod_j w_{a_j}
 (u_{a_1}\wedge v_{a_1})\wedge\cdots\wedge
 (u_{a_M}\wedge v_{a_M})
 =\frac1{M!}\Omega_F^{\wedge M}.
\]

The state can contain `binom(r,M)` Slater terms although the displayed bond is
only `r`. For `M>1` and generic `r>M`, it is non-decomposable and therefore not
the `chi=1` family.

The channel factorization is redundant: the physical state depends only on
`F`. Solver parameters should therefore use the skew matrix directly unless a
low pair-rank factorization is intentionally imposed.

## Polynomial overlap theorem

For two pair matrices `F,G`, define the formal generating function

\[
 Z_{F,G}(t)=\det(I+tF^\dagger G)^{1/2},
\]

using the square-root branch with constant term one. Pfaffian Cauchy--Binet
gives

\[
 Z_{F,G}(t)=\sum_{m\ge0}t^m
 \langle\Psi_m(F)|\Psi_m(G)\rangle.
\]

Thus the fixed-`M` overlap is one polynomial coefficient rather than a sum over
`binom(r,M)^2` path pairs.

Writing `A=F^dagger G` and `Z(t)=sum_m z_m t^m`, expansion of
`(1/2) Tr log(I+tA)` yields the trace--Newton recurrence

\[
 mz_m=\frac12\sum_{k=1}^m(-1)^{k+1}
 \operatorname{Tr}(A^k)z_{m-k},\qquad z_0=1.
\]

With dense matrices this costs

\[
 T_{\rm overlap}=O(MD^3),\qquad M_{\rm overlap}=O(D^2+M).
\]

The implementation keeps only one rolling matrix power, so the memory bound is
realized rather than inferred from parameter count.

## One- and two-body functional operators

For a one-body functional matrix `h`, transform the ket pair matrix by

\[
 G(t)=(I+th)G(I+th)^T.
\]

Then

\[
 \left.\frac{d}{dt}[t^M]Z_{F,G(t)}(t)\right|_{t=0}
 =\langle\Psi_M(F)|\sum_i h(i)|\Psi_M(G)\rangle.
\]

The code differentiates the trace recurrence analytically, preserving the
`O(MD^3)` asymptotic cost.

For a symmetrized factorized pair operator

\[
 V=\sum_{\ell=1}^L\frac{q_\ell}{2}
 (A_\ell\otimes B_\ell+B_\ell\otimes A_\ell),
\]

the mixed derivative of
`G(t,u)=(I+tA_l+uB_l)G(I+tA_l+uB_l)^T` generates the distinct-particle sum.
This gives

\[
 T_{2\rm b}=O(LMD^3)
\]

per bra/ket pair. A general exchange-symmetric two-body tensor has an operator
Schmidt decomposition with `L<=D^2`, giving a polynomial worst-case bound
`O(MD^5)`; density fitting or separable functional interactions can make `L`
much smaller.

## Systematic improvement by finite AGP sums

For

\[
 \Psi=\sum_{a=1}^K c_a\Psi_M(F_a),
\]

norm and operator expectations use `K^2` transition contractions:

\[
 T_{\rm norm/1b}=O(K^2MD^3),\qquad
 T_{2\rm b}=O(K^2LMD^3).
\]

This hierarchy is systematically complete in principle. Every even-particle
Slater determinant is itself an AGP by pairing its occupied orbitals, so finite
AGP sums contain finite Slater sums and eventually span `Lambda^{2M}V_D`.
Worst-case `K` can still be exponential; the scientific question is whether
interacting continuum targets converge at modest `K`.

For fixed pair matrices, the implementation now exposes the complete
polynomial transition matrices

\[
 S_{ab}=\langle\Psi_M(F_a)|\Psi_M(F_b)\rangle,\qquad
 H_{ab}=\langle\Psi_M(F_a)|H|\Psi_M(F_b)\rangle.
\]

The amplitudes are therefore a linear variational problem rather than generic
optimizer parameters. A conditioned solver diagonalizes `S`, discards
eigenvalues below a declared relative/absolute threshold, whitens the retained
subspace, and solves its Hermitian eigenproblem. It reports retained rank,
discarded rank, retained condition number, and the generalized residual
`||Hc-ESc||`. This removes amplitude scale and near-linear-dependence gauges;
the nonlinear pair-matrix gauge remains.

## Odd particle number by a blocked orbital

For `N=2M+1`, introduce one unpaired functional orbital `u` and define

\[
 \Psi_M^{\rm block}(F,u)
 =u\wedge\frac{\Omega_F^{\wedge M}}{M!}.
\]

Its coefficient on an increasing odd support `I` is the Pfaffian of the even
augmented minor

\[
 \operatorname{pf}\begin{pmatrix}
 F_{I,I}&u_I\\-u_I^T&0
 \end{pmatrix}.
\]

This state has a direct matrix-wedge FEMPS: prepend a bond-one `u` core to the
ordered-channel even AGP cores. For contraction, append one auxiliary basis
vector `a` and form

\[
 \widetilde F=\begin{pmatrix}F&u\\-u^T&0\end{pmatrix}.
\]

The even augmented state splits into orthogonal auxiliary-occupation sectors,

\[
 \Psi_{M+1}(\widetilde F)
 =\Psi_{M+1}(F)+\Psi_M^{\rm block}(F,u)\wedge a.
\]

Consequently, blocked overlaps are the augmented even overlap minus the purely
physical `M+1`-pair overlap. Extending every physical operator by a zero row and
column for `a` gives the same subtraction for one- and factorized two-body
matrix elements. The dense cost remains
`O((M+1)(D+1)^3)` for overlap/one-body and
`O(L(M+1)(D+1)^3)` for operator-Schmidt rank `L` two-body terms. No occupation
lattice or explicit antisymmetric tensor is used by the production route.

## Validation and numerical stability

The following independent constructions agree for deterministic complex tests:

- recursive Pfaffian minors;
- explicit normalized particle tensors;
- ordered-channel matrix-wedge FEMPS cores;
- exterior-coordinate dynamic programming;
- trace--Newton generating functions.

Norm, one-body, factorized two-body, finite-AGP-sum transitions, and their AD
gradients agree with explicit tensors. CPU/Blackwell normalized-energy and
gradient parity is also verified.

For the blocked extension, independent `D=5,N=3` Pfaffian minors, ordered FEMPS
cores, explicit particle tensors, auxiliary-sector contractions, and gradients
agree at worst to `1.14e-14`. At `D=32,N=21`, CPU/Blackwell normalized one-body
energy differs by `5.08e-11`, and both pair and blocked-orbital gradients agree
within `5.9e-12`.

The generic transition recurrence now scales the bra and ket separately by
detached maximum-entry magnitudes before forming `F^dagger G`, then restores
the exact homogeneous factor. This preserves reverse-mode derivatives and, in
the stress test, evaluates an unchanged transition when the two input scales
are `1e-120` and `1e120` with `3.6e-15` relative error.

Self norms use a stronger positive recurrence. A complex skew matrix has
pairwise-degenerate singular values `s_a,s_a`; therefore

\[
 \langle\Psi_M(F)|\Psi_M(F)\rangle
 =e_M(s_1^2,\ldots,s_{\lfloor D/2\rfloor}^2),
\]

where `e_M` is an elementary symmetric polynomial. Averaging each numerical
singular-value pair and accumulating positive coefficients eliminates the
alternating trace cancellation. `agp_log_norm` applies the same construction in
the log domain when the ordinary norm is outside floating-point range.

For dense real `D=64,N=64`, the former Newton norm has the wrong sign and
`2.57e17` relative error; the positive recurrence has `1.93e-14` relative
error. Across pair-matrix scales `1e-12` through `1e12`, the log norm agrees
with a `slogdet` top-sector truth within `2.28e-13`. This stabilization is exact
for self norms. Cross overlaps close to a true cancellation, and finite AGP
sums formed from them, remain intrinsically relative-condition limited.

## Scope limits

- A single blocked orbital covers odd particle number. Multiple blocked
  quasiparticles and spin-coupled blocking are not yet implemented.
- AGP is a restricted FEMPS variety, not a proof of generic matrix-wedge
  contractibility.
- Pair rank and AGP-sum length are expressivity controls, not yet a proven
  correlation entropy or canonical FEMPS spectrum.
- Representation advantage must be established on E1--E4 physics benchmarks,
  not inferred from the exponential formal Slater expansion alone.
- Every even matrix-wedge FEMPS also has an exact pathwise LC-AGP expansion,
  and every finite-basis even state is LC-AGP at sufficiently large K. See
  `lc_agp_relation.md`. The relevant open distinction is succinct polynomial
  contraction without enumerating that expansion, not membership in a
  “non-LC-AGP” state set.
