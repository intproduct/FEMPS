# Phase 18 report: multiscale half-line basis, structured MPO, and Gate F

## Outcome

Gate F is **PASS (controlled N=8 point, qualified by one auxiliary audit)**.
The core Phase 18 exit criteria pass:

1. a boundary-compatible two-scale odd-Hermite basis reduces the dominant
   N=6 basis error by `81.5%` relative to Gate E;
2. the production Fourier MPO is built and compressed incrementally, without
   materializing dense raw `(4M)^2 D^2` bulk tensors;
3. a blind `N=8,D=10` native MPS/MPO point meets the declared `1.2e-2`
   agreement budget against the exterior `D=12` numerical reference; and
4. an independent local optimizer agrees with the best blind AD point within
   `5.11e-5`, below the declared `5e-4` optimization budget.

One extra predeclared diagnostic does not pass: the bond-128 versus bond-192
raw MPS-parameter gradient relative difference is `2.90e-5`, above the
additional `2e-6` threshold. This is retained as a failed auxiliary check.
The corresponding fixed-state energy difference is `3.60e-7`, the maximum
gradient-component difference is `2.22e-7`, and gradient cosine similarity is
`0.999999999584`. The failed auxiliary threshold is not silently relaxed.

The result admits a controlled N=8 point. It is not an asymptotic scaling or
continuum-error theorem, and it does not admit chi-32 local DMRG on the current
GPU/contraction path.

## Two-scale collision-compatible basis

For local order `D`, the primitive set combines odd-Hermite half-line
functions at scales

```text
ell_short = ell/sqrt(rho),
ell_long  = ell*sqrt(rho),                 rho > 1.
```

The first `ceil(D/2)` odd modes use the short scale and the remaining
`floor(D/2)` modes use the long scale. Every primitive vanishes linearly at
the collision face. Analytic Gaussian-polynomial moments produce the overlap,
first derivative, negative second derivative, position, and squared-position
matrices. Symmetric Lowdin orthonormalization produces the final basis.

An independent 700-point half-line Gauss--Legendre audit gives maximum
absolute residuals

```text
overlap             3.03e-14
first derivative    3.58e-09
negative d2/dr2     5.98e-08
position            1.69e-14
position squared    8.08e-14
```

The collision-boundary residual is exactly zero in float64. At the production
`D=10,rho=2.5` choice the primitive overlap condition number is `9.65e4`;
at `D=12` it is `2.10e6`. These values are reported as conditioning controls,
not hidden by the final orthonormalization.

## Matched N=2 and N=4 basis controls

The N=2 reference is the independent relative-coordinate continuum value
`2.553831733979`. The common Phase 18 Fourier/local projection controls
(`M=96,Q=160`) leave an approximately `2e-7` operator floor at the highest
orders, so the D=8 and D=10 comparison should not be interpreted below that
scale.

| `D` | Best odd-Hermite error | Best multiscale error | Reduction |
|---:|---:|---:|---:|
| 4 | `4.351e-4` | `6.486e-6` | `98.5%` |
| 6 | `2.901e-6` | `5.294e-7` | `81.8%` |
| 8 | `2.411e-7` | `2.117e-7` | `12.2%` |
| 10 | `2.140e-7` | `2.018e-7` | `5.7%` |

The N=4 comparison uses the exterior `D=14` numerical reference
`11.0230828537`, which is not a continuum bound.

| `D` | Best odd-Hermite error | Best multiscale error | Reduction |
|---:|---:|---:|---:|
| 4 | `1.857e-2` | `1.110e-2` | `40.2%` |
| 6 | `6.605e-3` | `2.044e-3` | `69.1%` |
| 8 | `3.023e-3` | `8.977e-4` | `70.3%` |

The N=4 optima move from scales `0.9,0.8,0.7` for the single-scale basis to
`(ell,rho)=(0.7,2.0),(0.6,3.0),(0.6,3.0)`. The scale ratio is therefore an
explicit variational/numerical control, not a fixed hidden tuning constant.

## Incremental structured Fourier MPO

Gate E built the exact four-real-state recurrence and then temporarily stored
the dense raw direct-sum MPO before left-SVD compression. Phase 18 applies the
incoming retained transfer directly to the sparse recurrence at each site:

```text
[retained x (4M)] transfer
        -> sparse [1,c,s,T] recurrence contraction
        -> [retained x (4M) x D x D] intermediate
        -> local Hilbert--Schmidt SVD
        -> next retained transfer.
```

This is algebraically the same left-to-right compression as the old path. At
`N=3,4,5`, `D=3,M=24`, the incrementally built and raw-then-compressed dense
operators agree with relative Frobenius errors
`3.91e-16,1.10e-15,1.59e-15`. Their retained ranks and local discarded norms
also agree to roundoff.

The resource reduction is substantial:

| Point | Theoretical raw entries | Largest build intermediate | Reduction | Stored compressed entries |
|---|---:|---:|---:|---:|
| N=6,D=8,W=96 | 43,718,528 | 2,537,472 | `94.2%` | 2,172,928 |
| N=8,D=10,W=128 | 109,482,800 | 5,465,600 | `95.0%` | 9,494,800 |

The diagnostic flag `dense_raw_fourier_bulk_materialized` is false for every
production run. At `N=6,D=10`, a fixed one-million-dimensional global action
comparison between MPO bonds 128 and 192 has relative difference `8.60e-13`
and maximum absolute difference `4.05e-9`. Local discarded singular values
remain diagnostics, not global certificates.

## N=6 basis and optimization audit

All scale, capacity, and multi-seed choices precede construction of same-basis
truth. The blind D=8 scan selects `(ell,rho)=(0.50,2.5)`. Matrix-free CPU
Lanczos then gives

| Basis | Galerkin energy | Error vs exterior D=12 reference | Matvec calls | Residual |
|---:|---:|---:|---:|---:|
| D=8 multiscale | `25.0534947768` | `4.128e-3` | 241 | `8.89e-7` |
| D=10 multiscale | `25.0518179918` | `2.452e-3` | 101 | `1.02e-6` |

Gate E's single-scale D=8 error was `1.328e-2`. Thus multiscale D=8 reduces
the error by `68.9%`, and D=10 reduces it by `81.5%`. The reference
`25.0493664161` remains a numerical exterior D=12 value, not a continuum
bound.

Production uses `(D,M,Q,W,chi)=(10,96,192,128,32)` and the fixed four-stage
Adam schedule `300@1e-2, 500@3e-3, 500@1e-3, 300@3e-4`.

| Seed | Final energy | Error vs same-basis truth | Peak GPU memory |
|---:|---:|---:|---:|
| 1831 | `25.0518460031` | `2.801e-5` | 424.2 MB |
| 1832 | `25.0518368156` | `1.882e-5` | 424.7 MB |
| 1833 | `25.0518505061` | `3.251e-5` | 425.1 MB |

All seeds pass the declared `2e-3` optimization tolerance by a wide margin.

## Blind N=8 admission

The predeclared eight-point D=8 scale/ratio scan selects
`(ell,rho)=(0.50,2.5)` before the exterior reference is evaluated. Training
again uses `M=96`, while D=10 uses `Q=192`, MPO bond 128, MPS bond 32, and the
same four-stage schedule as N=6. No `10^8`-entry product-basis state is
materialized.

| Seed | Final energy | Error vs exterior D=12 reference | Peak GPU memory |
|---:|---:|---:|---:|
| 1861 | `44.4543787461` | `8.369e-3` | 533.3 MB |
| 1862 | `44.4543732636` | `8.364e-3` | 534.1 MB |
| 1863 | `44.4544087544` | `8.399e-3` | 534.8 MB |

The best blind error is `8.364e-3`, below the declared `1.2e-2` reference
agreement budget. The exterior D=12 value `44.4460095284` is a numerical
reference and does not convert this into a continuum bound.

On the best fixed state, MPO bonds 128 and 192 give energies
`44.4543732636` and `44.4543729039`. Their `3.60e-7` difference passes the
declared `1e-6` energy budget. The stricter auxiliary gradient result is the
qualified failure described in the outcome.

## Independent local optimizer and resource boundary

latticeTN's local Lanczos previously started from a random vector even when a
good two-site tensor was already available. With a small iteration budget this
could destroy a trained state. Phase 18 adds an optional Lanczos initial vector
and makes DMRG use the current two-site tensor by default. A direct N=8,D=8
smoke moves `44.4584886` to `44.4580095` instead of worsening it.

At N=8,D=10, chi-32 local DMRG is not admitted: the present effective-
Hamiltonian contraction requests a `78.12 GiB` CUDA intermediate on the
`23.89 GiB` RTX PRO 4000. This resource rejection is recorded explicitly.
For an independent bounded audit, the best chi-32 AD state is first brought to
exact right-canonical form and Schmidt-truncated to chi 16. The total sequential
discarded weight is `2.95e-8`, and the compressed energy is
`44.4543704357`. Two matrix-free local-Lanczos sweeps give
`44.4543221557`; the difference from the source chi-32 AD energy is
`5.11e-5`, below the declared `5e-4` optimizer-consistency budget. The two
sweep energies agree within `4.7e-9`, with maximum local truncation weight
`1.22e-9`.

This separates MPS capacity and optimizer error at the admitted chi-16 local
audit, but it does not claim that the current chi-32 DMRG contraction scales.

## Accuracy-to-resource trend and limits

The best controlled errors against the available numerical references are
approximately `2.0e-7` at N=2,D=10, `9.0e-4` at N=4,D=8,
`2.45e-3` at N=6,D=10, and `8.36e-3` at N=8,D=10. Basis error still grows
with particle number at fixed local order. The N=8 point is admitted by its
explicit budget, not by extrapolating an asymptotic convergence law.

The production MPO construction problem from Gate E is removed, but three
limitations remain:

1. the N=8 exterior reference is numerical rather than a continuum bound;
2. bond-128 raw parameter gradients miss an intentionally strict auxiliary
   bond-convergence threshold, despite near-identical direction and passing
   energy convergence; and
3. latticeTN's current local effective-Hamiltonian contraction has an
   unacceptable chi-32 intermediate.

These define Phase 19. N=10 and favorable asymptotic claims are not admitted.

## Prior-art and naming boundary

Hong et al. remain the parent for orthonormal functional bases, projected
operators, coefficient MPS, and global AD. Li--Waintal remain the parent for
ordered first-quantized distance MPS. The two-scale half-line basis,
incremental sparse-recurrence compression, and controlled N=8 evidence are an
implementation/integration contribution. No priority claim is attached to
ordered coordinates, first-quantized MPS, multiscale spectral bases in
general, or fermionic tensor networks. This route remains an ordered-distance
functional TN and is not called FEMPS.

## Reproduction

```powershell
.\.venv\Scripts\python scripts\benchmark_phase18_basis_and_mpo.py
.\.venv\Scripts\python scripts\benchmark_phase18_n6_n8.py --device auto
```

The second command writes a recoverable ignored checkpoint before the local
DMRG audit. Machine-readable records are
`results/phase18_basis_structured_controls.json` and
`results/phase18_multiscale_n6_n8.json`.

