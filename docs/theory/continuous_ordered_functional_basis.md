# Continuous ordered-distance functional basis

## Evidence level and purpose

This is the Phase 16--17 bridge from the finite gap grid to the 2201
architecture. The coordinate and operator identities are exact. Dirichlet
sine and odd-Hermite bases are Galerkin approximations; the interval polynomial
and Fourier--Bessel soft-Coulomb representations are controlled numerical
approximations. Gate E passes at controlled `N<=6` scope. Basis efficiency,
temporary dense raw-MPO storage, and N=8 accuracy remain open.

## Center of mass and positive distances

In the ordered chamber `x_0<...<x_(N-1)`, define

```text
R   = (1/N) sum_i x_i,
r_i = x_i-x_(i-1) > 0,       1 <= i <= N-1.
```

Writing `q=(R,r_1,...,r_(N-1))=B x`, the inverse `x=A q` has

```text
A[k,0] = 1,
A[k,i] = i/N             if i <= k,
       = -(N-i)/N        if i > k.
```

The Jacobian has absolute determinant one.  A unit-normalized full
antisymmetric wavefunction maps to `sqrt(N!)` times its restriction to this
chamber; collision faces `r_i=0` are Dirichlet boundaries.

The kinetic metric `G=B B^T` separates the center of mass:

```text
G[0,0] = 1/N,
G[0,i] = 0,
G[i,j] = 2 delta_ij - delta_(i,j+1) - delta_(i+1,j).
```

Thus

```text
T = -1/(2N) d_R^2
    - sum_i d_(r_i)^2
    + sum_i d_(r_i) d_(r_(i+1)).
```

For the unit harmonic trap, `sum_i x_i^2/2=q^T K q/2` with

```text
K[0,0] = N,
K[0,i] = 0,
K[i,j] = min(i,j) [N-max(i,j)] / N.
```

All formulas are checked for `N=1,...,6` against the original coordinates,
matrix inverses, Jacobians, and an independent Hessian chain rule.

## Local functional bases

The center-of-mass site uses the existing full-line harmonic-oscillator basis,
scaled by its natural length `1/sqrt(N)`.  The first half-line candidate uses

```text
phi_n(r) = sqrt(2/Rmax) sin[(n+1) pi r/Rmax],
0 < r < Rmax.
```

It enforces both the collision boundary and a separately controllable outer
box boundary.  The overlap, first derivative, negative second derivative,
position, and position-squared matrices are analytic.  Higher affine powers
use Gauss--Legendre projection with an independent quadrature-order control.
Tests compare every matrix against separate quadrature.

The sine basis is not asserted to be optimal. Its algebraic convergence at a
non-smooth chamber corner motivated an implemented unbounded comparison. The
second candidate is the normalized restriction of odd harmonic-oscillator
functions to `r>0`, scaled by a positive length `ell`. It is orthonormal on the
half-line, vanishes at collision, and has no outer box. Its overlap, `r^2`, and
`-d^2/dr^2` matrices are analytic; `r` and the first derivative are checked by
independent quadrature. Phase 17 supports this unbounded basis with projected
characteristic operators and a Fourier--Bessel interaction. The finite sine
construction remains an independent matched-order control.

## Native noninteracting MPO

The Galerkin Hamiltonian is a sum of one-site kinetic/trap terms, adjacent
two-site derivative products, and all relative-coordinate quadratic products.
For `N` sites its present direct-sum MPO has at most

```text
W_0 = 2N + (N-2) + binom(N-1,2)
    = (N^2+3N-2)/2
```

channels.  N=2 separates exactly into center-of-mass and relative operators.
N=3 explicitly exercises the mixed derivative and is Hermitian to roundoff.

For noninteracting spinless fermions the exact continuum energy is `N^2/2`.
At fixed `Rmax=6`, the N=3 errors for sine orders `4,6,8,10` are
`8.53e-2,1.79e-2,6.40e-3,3.09e-3`. With odd Hermite scale `ell=0.7`, orders
`6,8,10` give `1.17e-2,1.65e-3,6.09e-4`; at order eight the independent scale
scan has a broad minimum around `ell=0.7--0.8`. These are numerical evidence
of improved basis efficiency, not a convergence-rate theorem.

## Soft-Coulomb interval polynomial

For particles `i<j`,

```text
x_j-x_i = sum_(a=i+1)^j r_a.
```

On the finite interval `0<=s<=S=(j-i)Rmax`, approximate

```text
1/sqrt(s^2+epsilon^2) ~= sum_(p=0)^K c_p t^p,
t = 2s/S-1.
```

The first distance site contributes `2r/S-1` and later sites contribute
`2r/S`, so their sum is exactly `t`.  The binomial transition

```text
M[p,q](z) = binom(q,p) z^(q-p),       q >= p
```

propagates every power of the partial sum.  One pair therefore has bond
`K+1`; a direct sum over pairs gives conservative bond `O(N^2 K)`, independent
of the local basis order.  Projected local powers are calculated before MPO
assembly, avoiding the false identity obtained by powering a truncated
position matrix.

Three errors remain separate:

1. sampled scalar Chebyshev error (not an interval-arithmetic certificate);
2. Gauss--Legendre error in the projected power matrices; and
3. functional basis/outer-box error.

Degrees `8,16,24` on `s in [0,12]` have sampled scalar errors decreasing to
below `2e-5`.  One- and two-distance projected MPOs independently match direct
one- and two-dimensional quadrature, with error decreasing in `K`.

## Soft-Coulomb Fourier--Bessel recurrence

On the unbounded half-line, use the exact cosine transform

```text
1/sqrt(s^2+a^2) = (2/pi) integral_0^inf K0(a k) cos(k s) dk.
```

The numerical rule has independent Fourier order, dimensionless frequency
cutoff, and odd-Hermite projection quadrature. The map `k=kmax*u^2` regularizes
the logarithmic endpoint of `K0` before Gauss--Legendre quadrature. Projected
`C=cos(k r)` and `S=sin(k r)` matrices pass independent one- and two-gap
half-line checks.

For one Fourier node, a row state `[1,c,s,T]` crossing a gap updates

```text
c' = C(c+1)-S s,
s' = S(c+1)+C s,
T' = T+c'.
```

Selecting `T` after the final gap yields every pair cosine. The all-pair
interaction therefore has raw bond `4M`, independent of N. Direct-pair and
compact operators agree globally to float64 precision. Compression is admitted
only after a bounded dense-operator or global-action audit; local discarded
singular values are not certificates.

## Controlled physics evidence

For N=2 soft Coulomb, `(D,Rmax,K)=(12,9,20)` gives
`2.5538326754`.  An independent half-line finite-difference Richardson value
differs by less than `2e-6`.  Three random Blackwell runs with full bond 12 end
near `2.5538355`, about `3e-6` above their Galerkin truth, using 288 MPS
parameters and no product-state gather.

For N=4 soft Coulomb, an independent matrix-free MPO/Lanczos audit at
`(D,Rmax,K)=(10,4.5,20)` gives the finite-Galerkin energy `11.0274291400`.
Orders `6,8,10,12` at fixed `Rmax=4.5` give
`11.0457788,11.0317439,11.0274291,11.0256076`, while the independent exterior
`D=14` numerical reference is `11.0230829`. At fixed `D=10`, boxes
`3.5,4.0,4.5,5.0` give `11.0413285,11.0274422,11.0274291,11.0291167`, exposing
both the small-box wall and the large-box resolution error.

At the production point, changing the interaction degree from `K=20` to
`K=24` shifts the N=4 Galerkin energy by `8.01e-7`. TT-SVD of the independent
Galerkin ground state gives energy errors `2.10e-2,1.47e-5,3.44e-9` at maximum
MPS bonds `2,4,8`; bond 16 is exact to the Lanczos residual. Thus the formal
bond-32 training point is not representation limited.

Three blind N=4 Blackwell runs finish `2.84e-5--4.63e-5` above their independent
finite-Galerkin truth and `4.37e-3--4.39e-3` above the exterior `D=14`
reference. Three N=2 runs finish within `3.01e-6` of their post-training
Galerkin truth. CPU and GPU energy agree exactly for the parity state and the
largest gradient difference is `8.88e-15`. Training uses only native MPS/MPO
contractions and never materializes a product-basis state.

With the unbounded interaction, matched N=4,D=8 odd Hermite and sine errors
against the same exterior numerical reference are `3.023e-3` and `8.661e-3`.
At N=6,D=8, the scale-0.60 Galerkin energy is `25.0626429274`; three blind
bond-32 runs lie `1.29e-4--2.48e-4` above it. TT-SVD of that Galerkin ground
state has only `4.50e-6` energy error at bond eight, while the difference from
an exterior D=12 numerical reference is `1.328e-2`. The observed larger-system
error is therefore basis dominated.

## Gate E decision

Gate E is **PASS (controlled unbounded N=6 prototype)**. The coordinate map,
signed recovery, collision boundary, and chamber normalization remain exact.
The interacting odd-Hermite basis improves the finite sine box at matched N=2
and N=4 orders. Every adopted MPO compression has a global audit, and three
blind N=6 runs finish within `2.49e-4` of post-run same-basis Galerkin truth
without a product-state gather.

The scope remains narrow. The N=6,D=8 Galerkin energy differs from an exterior
D=12 numerical reference by `1.328e-2`; TT-SVD shows this is basis dominated,
not an MPS-capacity limitation. N=8, continuum convergence rates, and favorable
end-to-end asymptotic resource requirements are not inferred.
