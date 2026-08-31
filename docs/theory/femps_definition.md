# FEMPS definition and small-system algebra

## 1. Exterior Hilbert-space convention

Let the field be `F=R` or `C`, and let the one-particle space `V_D` have an
orthonormal basis `e_1,...,e_D`. We give `Lambda^p V_D` the inner product for
which

\[
 e_{i_1}\wedge\cdots\wedge e_{i_p},\qquad i_1<\cdots<i_p,
\]

is an orthonormal basis. Its isometric embedding into `V_D^{tensor p}` is

\[
 J_p(v_1\wedge\cdots\wedge v_p)
 =\frac1{\sqrt{p!}}\sum_{\pi\in S_p}\operatorname{sgn}(\pi)
 v_{\pi(1)}\otimes\cdots\otimes v_{\pi(p)}.
\]

Accordingly, explicit tensor multiplication uses

\[
 J_{p+q}(\omega\wedge\eta)
 =\sqrt{\frac{(p+q)!}{p!q!}}\,
 \operatorname{Alt}_{p+q}\bigl(J_p(\omega)\otimes J_q(\eta)\bigr).
\]

This convention makes an orthonormal-orbital Slater determinant have norm one.
Spin and other internal labels are included in `V_D`; the exterior product
antisymmetrizes the entire one-particle label.

## 2. Open-boundary matrix-wedge state

For sites `j=1,...,N`, choose

\[
 A^{[j]}\in
 \operatorname{Mat}_{\chi_{j-1}\times\chi_j}(V_D),
 \qquad \chi_0=\chi_N=1.
\]

For a matrix of `p`-forms `X` and a compatible matrix of `q`-forms `Y`, define

\[
 (X\wedge Y)_{ac}=\sum_b X_{ab}\wedge Y_{bc}.
\]

The open-boundary FEMPS is

\[
 C(A^{[1]},\ldots,A^{[N]})
 =\left[A^{[1]}\wedge\cdots\wedge A^{[N]}\right]_{11}
 \in\Lambda^N V_D.
\]

Equivalently it is the sum, over every virtual path, of the wedge of the `N`
one-forms encountered along that path. This equivalence is used as an
independent executable oracle and explicitly costs the product of internal
bond dimensions.

### Proposition 1: associativity and strict antisymmetry

Matrix-wedge multiplication is associative. Indeed, bilinearity and exterior
associativity give

\[
 ((X\wedge Y)\wedge Z)_{ad}
 =\sum_{b,c}(X_{ab}\wedge Y_{bc})\wedge Z_{cd}
 =\sum_{b,c}X_{ab}\wedge(Y_{bc}\wedge Z_{cd}).
\]

Every path term and hence their sum lies in `Lambda^N V_D`, so the state is
strictly alternating without a penalty or post-projection. Matrix-wedge is not
generally graded commutative because the matrix indices carry an order.

## 3. Expressivity statements

### Proposition 2: the `chi=1` family

If all bonds equal one, then

\[
 C=a^{[1]}\wedge\cdots\wedge a^{[N]},
\]

which is decomposable. Conversely, every decomposable `N`-form has such a
representation. Thus nonzero normalized `chi=1` states are precisely single
Slater determinants after choosing normalized orbitals; dependent one-forms
give the zero state.

The FEMPS correlation multiplicity is therefore one for a single Slater,
whereas its ordinary particle-TT ranks remain `{N choose k}`. This is a change
of representation category, not a reduction of the same Schmidt rank.

### Proposition 3: finite Slater sums

Every weighted sum of `R` decomposable terms embeds with all internal bonds at
most `R`. Assign one diagonal virtual path to each term and put its scalar
weight on any one core. Off-diagonal core entries vanish. Therefore

\[
 \chi_j^{\mathrm{FEMPS}}\le R
\]

for this construction. This is an upper bound only: virtual-path interference
can make the minimal FEMPS bonds smaller than a displayed Slater-sum length.

### Proposition 4: exact `N=2` characterization

For two particles,

\[
 C=\sum_{a=1}^{\chi}u_a\wedge v_a.
\]

Let `K` be its skew coefficient matrix. Each summand contributes
`u_a v_a^T-v_a u_a^T`, so `rank(K)<=2 chi`. Conversely, the standard congruence
normal form of a real or complex skew matrix of rank `2r` writes it as a sum of
`r` such rank-two blocks. Hence

\[
 \chi_{\min}^{\mathrm{FEMPS}}(C)=\frac12\operatorname{rank}K.
\]

Thus the `N=2` ansatz covers all bivectors, and its minimal bond is exactly the
decomposable/exterior rank. Multiplying `K` by the normalization factor used in
the full particle tensor does not change this rank.

## 4. Gauge action

Let `G_j in GL(chi_j,F)` and set `G_0=G_N=1`. The usual scalar virtual-bond
action

\[
 \widetilde A^{[j]}=G_{j-1}^{-1}A^{[j]}G_j
\]

leaves the FEMPS invariant: scalar gauge entries commute with exterior forms,
and neighboring `G_j G_j^{-1}` factors cancel after the virtual sums. This
establishes the ordinary MPS gauge subgroup.

It does **not** yet establish a canonical form. A canonicalization based on the
Euclidean norm of individual cores is not justified because the physical norm
is an exterior/determinant contraction. Possible extra redundancies,
gauge-independent minimal bonds, and a correlation-multiplicity spectrum remain
open until the contraction structure is understood.

## 5. Computational status

The parameter count of displayed cores is `sum_j D chi_{j-1} chi_j`, but this
does not imply polynomial contraction. Both current exact implementations are
exponential:

1. matrix-wedge multiplication materializes `D^N` antisymmetric tensors;
2. path enumeration sums `prod_j chi_j` Slater determinants.

They agree for deterministic `N=2,3,4` tests and are Phase 2 truth oracles only.
No scalable-solver claim is made before Gate A.

## 6. Open algebraic questions

- Minimal FEMPS bonds versus Slater/secant rank for `N>=3`.
- Dimension after quotienting generic core parameters by the gauge group.
- Additional exterior-specific gauge redundancies and canonical forms.
- A gauge-independent statistics-carrier/correlation-multiplicity split.
- Pathological low-Slater-rank families with inefficient non-diagonal FEMPS
  descriptions, if any.
