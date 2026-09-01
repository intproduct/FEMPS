# Controlled approximate exterior contraction gate

## Status and scope

This note closes Candidate K1 for a **generic** polynomial-cost approximate
FEMPS contraction.  It does not reject promised positive, well-conditioned, or
otherwise structured subclasses.  Strict antisymmetry is preserved throughout:
the state is still defined in the exterior algebra; only scalar contractions
are approximated.

The obstruction already appears in the Phase 22 upper-bidiagonal path family,
so it applies before one- or two-body observable machinery is considered.

## Error targets

Let

```text
n = <Psi|Psi> >= 0,
h = Re <Psi|H|Psi>,
E = h/n                         when n > 0.
```

For a randomized norm contraction, a relative `(epsilon,delta)` guarantee is

```text
Pr[(1-epsilon)n <= n_tilde <= (1+epsilon)n] >= 1-delta,
0 < epsilon < 1.
```

This definition forces a zero output on a zero-norm input.  An additive
guarantee instead has

```text
Pr[|n_tilde-n| <= Delta_n] >= 1-delta.
```

It is useful for energy only when the same run certifies
`n_tilde > Delta_n`.  A state-vector approximation must separately specify a
norm such as `||Psi-Phi|| <= eta`; a low selected term count or coefficientwise
fit is not itself a state error bound.

## Theorem 1: PSD-permanent transfer to squared norm

Assume there is a polynomial-time randomized relative approximation scheme for
the squared norm of every real upper-bidiagonal matrix-pair input, polynomial
jointly in `(M,D,w,1/epsilon,log(1/delta))` and in the coefficient bit length.
Then there is a PRAS for the permanent of a real PSD matrix.

### Proof

For a real PSD `M x M` matrix `A`, take `D=2M`, the orthonormal pair forms

```text
P_j = e_(2j-1) wedge e_(2j),
F_i = sum_j A_(i,j) P_j,
```

and the Phase 22 unique path with `w=M+1`.  Its normalized state is

```text
Psi_A = perm(A) P_1 wedge ... wedge P_M / M!,
n_A = perm(A)^2 / (M!)^2.
```

The permanent of a real PSD matrix is nonnegative.  Given a relative norm
estimate `n_tilde`, output

```text
p_tilde = M! sqrt(n_tilde).
```

On the success event,

```text
sqrt(1-epsilon) perm(A)
  <= p_tilde
  <= sqrt(1+epsilon) perm(A).
```

Choosing the norm tolerance polynomially from any requested permanent
tolerance gives a PRAS.  Construction size and the bit length of `M!` are
polynomial.  Meiburg's theorem excludes such a PRAS for real PSD permanents
unless `RP=NP` [@Meiburg2023PSDPermanent].  Hence the assumed generic squared-
norm scheme would imply `RP=NP`.  QED.

The later exponential inapproximability result for Hermitian PSD permanents
strengthens the separation between current simply exponential algorithms and
relative approximation [@EbrahimnejadNagdaOveisGharan2025PSDPermanent].  It is
not needed for the transfer theorem.

### Why the nonnegative FPRAS does not contradict the theorem

Jerrum--Sinclair--Vigoda cover matrices whose **entries** are nonnegative
[@JerrumSinclairVigoda2004PermanentFPRAS].  A real PSD matrix can have negative
off-diagonal entries.  Neither promise contains the other in general.  The
FPRAS therefore proves a tractable special cone, while the real-PSD reduction
rejects an algorithm advertised for arbitrary FEMPS pair coefficients.

Moreover, entrywise nonnegativity depends on the declared paired-orbital and
geminal gauges.  A future positive-cone method must either freeze that gauge or
prove its promise is maintained by every optimization update.

## Proposition 2: additive permanent error does not control the norm uniformly

Gurvits's estimator applies to arbitrary complex matrices with additive scale
`epsilon ||A||^M` [@AaronsonHance2014Gurvits].  No uniform conversion from that
bound to relative squared-norm error is possible.

For example, with rational `tau > 0`,

```text
A_tau = [[1, 1], [1, -1+tau]],
perm(A_tau) = tau.
```

The input norm stays of order one while the exterior squared norm is
`tau^2/4`.  Taking `tau=2^-L` makes the relative accuracy needed of an additive
permanent estimator exponential in the input precision `L`; at `tau=0` the
exterior state is exactly zero.  The same conditioning issue appears for a
direct additive norm estimator whenever `Delta_n >= n`.

This proposition does not say that additive estimates are useless.  They are
admissible under a declared promise such as a polynomially representable lower
bound `n >= n_min > 0`, with cost polynomial also in the resulting condition
number.

## Lemma 3: certified Rayleigh-quotient propagation

Suppose deterministic bounds, or simultaneous confidence events, give

```text
|n-n_tilde| <= Delta_n,
|h-h_tilde| <= Delta_h,
n_tilde > Delta_n,
E_tilde = h_tilde/n_tilde.
```

Then

```text
|E-E_tilde|
  <= (Delta_h + |E_tilde| Delta_n)/(n_tilde-Delta_n).       (1)
```

Indeed,

```text
E-E_tilde
 = ((h-h_tilde) - E_tilde (n-n_tilde))/n,
```

and `n >= n_tilde-Delta_n > 0`.  If the two scalar bounds fail with
probabilities at most `delta_n` and `delta_h`, respectively, (1) holds with
probability at least `1-delta_n-delta_h`, without assuming independence.

An exact interval is also available.  Let

```text
H = [h_tilde-Delta_h, h_tilde+Delta_h],
N = [n_tilde-Delta_n, n_tilde+Delta_n] subset (0,infinity).
```

The certified energy interval is the minimum and maximum of `x/y` over the
four endpoint pairs `(x,y) in endpoints(H) x endpoints(N)`.  A ratio of two
unbiased estimators is generally biased, so an empirical error bar on
`h_tilde/n_tilde` cannot replace this simultaneous interval.

For a bounded Hamiltonian, a certified state-vector approximation also gives a
route: if normalized states obey `||psi-phi|| <= eta`, then

```text
|<psi|H|psi>-<phi|H|phi|| <= 2 ||H|| eta.
```

The generic APG selection literature does not supply this bound for an
arbitrary matrix-pair input.

## Required controls and their outcome

The exact Phase 24 certificate contains four diagnostic classes:

| Class | Representative | What it establishes |
|---|---|---|
| positive | a nonnegative integer matrix | the JSV promise is nonempty and the exterior identity remains exact |
| cancelling | `A_0` above | a compact nonzero input can define the zero exterior state |
| ill-conditioned | `A_(2^-L)` | additive-to-relative conversion scales exponentially with input precision |
| real PSD, signed entries | `[[1,-1/2],[-1/2,1]]` | the admitted hard promise is not the nonnegative cone |

Existing physically optimized project points are finite LC-AGP states with
deterministic Pfaffian contraction.  They remain valuable positive controls,
but importing them here would test the already tractable baseline rather than
the generic APG/matrix-pair estimator.  No new “physically optimized” K1 run is
admitted after the norm gate fails; a future promised subclass must define its
physical optimizer and compare every small point with full exterior truth.

## Bias, variance, and resource contract for any successor

An approximate successor must report jointly:

1. whether norm and numerator estimators are unbiased, and the bias of the
   final ratio or interval construction;
2. a non-asymptotic variance or tail bound and total failure probability;
3. the conditioning parameter controlling `Delta_n/n`;
4. wall time and peak memory as functions of all physical and approximation
   parameters; and
5. exact exterior truth on positive, cancelling, ill-conditioned, signed/complex,
   and physically optimized small instances.

Tensor-network Monte Carlo demonstrates that unbiased stochastic contraction
can be useful [@Ferris2015TNMC], but its sampling distribution and variance do
not automatically transfer to exterior matrix multiplication.  The generic
fermion sign-problem theorem is only contextual prior art
[@TroyerWiese2005SignProblem]; Theorem 1 is the direct obstruction used here.

## Gate K decision

**Gate K: FAIL for generic Candidate K1.**

The required relative norm certificate would yield a PRAS for real PSD
permanents, while a generic additive guarantee cannot keep the Rayleigh
denominator away from zero.  Consequently no GPU/AD approximate matrix-pair
solver is admitted on the unrestricted family.

The following narrower directions remain logically open and must be presented
as promises, not generic FEMPS contraction:

- entrywise-nonnegative paired coefficients in a fixed gauge;
- a certified polynomial norm lower bound together with additive estimators;
- strongly orthogonal, low-rank, selected-pairing, Gaussian/Pfaffian, or other
  structural subclasses that explicitly exclude the PSD hard embedding; and
- heuristic stochastic contraction reported without a variational-energy
  guarantee, as an empirical comparator only.
