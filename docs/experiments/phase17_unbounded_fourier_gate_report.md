# Phase 17 report: unbounded Fourier--Bessel interaction and Gate E

## Outcome

Gate E is **PASS (controlled unbounded N=6 prototype)**. The finite sine-box
bottleneck is removed by combining the collision-Dirichlet odd-Hermite basis
with a Fourier--Bessel soft-Coulomb separation. A compact recurrence represents
all particle pairs with raw interaction bond `4M`, independent of particle
count. Every production compression has a bounded global operator/action
audit, and blind native GPU training reaches a controlled `N=6,D=8` point
without a product-basis gather.

The pass is not an asymptotic scalability or continuum-accuracy claim. At the
N=6 point, the observed `1.328e-2` difference from an exterior `D=12`
numerical reference is dominated by the local functional basis. Increasing
MPS bond is not the next accuracy bottleneck.

## Unbounded soft-Coulomb separation

The exact cosine transform is

```text
1/sqrt(s^2+a^2) = (2/pi) integral_0^inf K0(a k) cos(k s) dk.
```

It follows from the standard integral representation of `K0` and cosine
transform inversion; see [NIST DLMF 10.32.E6](https://dlmf.nist.gov/10.32.E6).
The numerical rule truncates at `a*k=30`, maps `k=kmax*u^2`, and applies
Gauss--Legendre quadrature in `u`. The square map regularizes the integrable
logarithmic endpoint behavior of `K0`. Fourier order, cutoff, and local
half-line projection quadrature remain independent controls.

On `0<=s<=12`, the maximum sampled scalar errors at
`M=64,96,128,160,192` are
`9.83e-3,1.55e-4,7.08e-8,2.89e-8,1.40e-8`. The nonmonotone improvement after
128 is reported rather than hidden; it reflects the fixed finite cutoff and
quadrature balance.

Odd-Hermite projections of `cos(k r)` and `sin(k r)` use a separate finite
Gauss--Legendre rule with a Gaussian-tail cutoff. Direct half-line quadrature
gives one-gap Frobenius errors `2.48e-6,4.95e-7,1.57e-7,6.47e-8` and direct
two-gap quadrature gives `5.76e-6,1.15e-6,3.66e-7,1.50e-7` at
`M=64,96,128,160`.

## Compact all-pair MPO

For one Fourier node, let `c,s` be the cosine/sine sums from particles left of
the current particle and let `T` accumulate all pair cosines. Crossing one
positive gap with projected multiplication operators `C=cos(k r)` and
`S=sin(k r)` updates

```text
c' = C (c+1) - S s,
s' = S (c+1) + C s,
T' = T + c'.
```

Thus a row state `[1,c,s,T]` requires four real channels per Fourier node.
Starting from `[1,0,0,0]` and selecting `T` after the last gap produces
`sum_(i<j) cos(k(x_j-x_i))`. The raw interaction bond is `4M`, independent of
`N`; the direct particle-pair construction remains available only as an audit.

At `N=4,D=3,M=24`, compact and direct-pair dense operators differ by
`5.17e-16` in relative Frobenius norm. At fixed `D=2,M=16`, the compact bond is
64 for `N=3,...,8`, whereas the direct-pair maximum bond grows
`34,68,134,201,300,400`. At N=8 the dense-block tensor element counts are
98,816 versus 2,325,192.

## Global compression audits

Local discarded singular values are recorded only as diagnostics and never as
global certificates.

For the full `N=4,D=4,M=64` Hamiltonian, the raw maximum bond is 269. Direct
dense-operator comparison gives relative Frobenius errors
`9.17e-2,2.48e-4,8.25e-6,5.87e-14` at compressed bonds `8,16,24,32`.

For `N=4,D=8,M=96`, the raw maximum bond is 397. A fixed 4,096-dimensional
global action audit gives relative errors `2.23e-6,3.21e-9,9.27e-15` at bonds
`32,48,64`.

For the production `N=6,D=8,M=96` point, the raw maximum bond is 413 and the
raw MPO has 43,718,528 scalar tensor entries. A fixed 262,144-dimensional
global action audit gives relative errors
`1.15e-6,1.63e-9,2.28e-14` at bonds `64,96,128`. Training therefore uses bond
96; the independent truth MPO uses bond 128.

## Matched finite-box and unbounded bases

The N=2 independent continuum reference is `2.553831733979`.

| `D` | Best odd-Hermite scale | Odd error | Sine box-9 error |
|---:|---:|---:|---:|
| 4 | 1.2 | `1.183e-5` | `3.540e-2` |
| 6 | 1.2 | `8.581e-7` | `2.308e-4` |
| 8 | 1.1 | `4.880e-8` | `7.648e-5` |
| 10 | 1.1 | `2.160e-8` | `7.744e-6` |
| 12 | 1.0 | `9.949e-9` | `9.391e-7` |

The N=4 comparison uses the exterior `D=14` numerical reference
`11.0230828537`.

| `D` | Best odd-Hermite scale | Odd error | Sine box-4.5 error |
|---:|---:|---:|---:|
| 4 | 0.9 | `1.857e-2` | `1.004e-1` |
| 6 | 0.8 | `6.605e-3` | `2.270e-2` |
| 8 | 0.7 | `3.023e-3` | `8.661e-3` |

The unbounded interacting basis therefore improves the finite-box basis at
every matched small-system order tested. At the N=4 production point,
`M=96` differs from `M=128` by `9.05e-7`, while local projection quadratures
96 through 224 change the energy by at most `3.10e-11`.

## Blind N=6 training and independent truth

All scale, MPS-bond, and multi-seed runs finish before the same-basis Galerkin
truth is constructed. A blind five-point scale scan selects `ell=0.60`.
Production uses `(D,M,Q,W,chi)=(8,96,160,96,32)`, 800 Adam steps, and only
native latticeTN contractions.

The post-run matrix-free Lanczos audit gives the same-basis ground energy
`25.0626429274`. It uses a 262,144-entry vector but never materializes the
squared Hamiltonian matrix.

| Seed | Final energy | Optimizer error | Fidelity | Time | Peak GPU memory |
|---:|---:|---:|---:|---:|---:|
| 1731 | `25.0628750749` | `2.321e-4` | `0.999983990` | 19.9 s | 736.7 MB |
| 1732 | `25.0627719158` | `1.290e-4` | `0.999994353` | 19.9 s | 737.0 MB |
| 1733 | `25.0628910209` | `2.481e-4` | `0.999986886` | 19.9 s | 737.3 MB |

All final norms are one to float64 precision. The peak includes raw MPO
construction and compression, not only optimization.

At selected scale 0.60, Galerkin orders `D=4,6,8` give
`25.2671,25.0924,25.0626429`, whose errors against the exterior `D=12`
numerical reference `25.0493664` are approximately
`2.18e-1,4.30e-2,1.328e-2`. This establishes convergence but also identifies
the basis as the dominant remaining error.

Post-training TT-SVD of the independent D=8 ground state separates MPS
capacity from optimizer behavior:

| Maximum MPS bond | Energy error | Fidelity |
|---:|---:|---:|
| 4 | `2.885e-3` | `0.999849620` |
| 8 | `4.503e-6` | `0.999999885` |
| 16 | `1.223e-8` | `0.999999999803` |
| 32 | `2.963e-12` | `0.999999999999958` |

The `chi=32` production choice is therefore not representation limited. The
roughly `1.3e-2` total discrepancy must not be attributed to insufficient MPS
bond.

## Gate E decision and limits

Gate E passes because:

1. the interacting unbounded basis improves the matched sine-box basis at
   controlled N=2 and N=4 points;
2. scalar, one-gap, two-gap, local projection, and energy convergence are
   independently measured;
3. the compact all-pair recurrence has measured constant-in-N Fourier bond and
   agrees globally with the direct-pair operator;
4. every adopted MPO compression has a global dense-operator or action audit;
5. blind N=6 native AD meets the declared `2e-3` optimizer tolerance for every
   seed, while the controlled total error meets the declared `2e-2` tolerance;
6. post-training Lanczos and TT-SVD separate basis, operator, MPS-capacity, and
   optimization errors.

The pass supports a controlled N=6 ordered-distance functional-TN solver. It
does not prove a favorable continuum accuracy-to-D law, does not cover N=8,
and does not remove the temporary dense raw-MPO construction cost. Those are
the next gate.

## Prior-art and naming boundary

Hong et al. remain the parent for orthonormal functional bases, operator
projection, coefficient MPS, and global AD. Li--Waintal remain the parent for
ordered first-quantized distance MPS and scalable distance-space solvers. The
Phase 17 contribution is the Fourier--Bessel/odd-Hermite integration, compact
real recurrence, and its controlled implementation evidence.

No priority claim is attached to the ordered chamber, distance variables, or
first-quantized MPS. This route remains an ordered-distance functional tensor
network and is not called FEMPS. Gate E does not change the conditional
obstruction for generic matrix-wedge FEMPS.

## Reproduction

```powershell
.\.venv\Scripts\python scripts\benchmark_ordered_continuous_fourier.py
.\.venv\Scripts\python scripts\benchmark_ordered_continuous_fourier_n6.py --device auto
```

Machine-readable records are
`results/phase17_unbounded_fourier_controls.json` and
`results/phase17_unbounded_fourier_n6.json`.
