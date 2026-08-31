# Phase 16 report: continuous ordered-distance Gate D

## Outcome

Gate D is **PASS (controlled continuous-basis prototype)**. The solver now
combines an exact ordered-sector fermion map with the defining machinery of the
2201 functional tensor network: orthonormal local functions, projected
continuous operators, native MPS/MPO expectations, and global automatic
differentiation. The pass is restricted to controlled `N<=4` problems and is
numerical evidence, not a large-particle or novelty claim.

## Exact fermionic and coordinate structure

In the chamber `x0<...<x_(N-1)`, the production variables are the center of
mass `R` and positive gaps `ri=x_i-x_(i-1)`. The forward and inverse maps have
Jacobian magnitude one. The kinetic metric is a decoupled center-of-mass entry
plus the tridiagonal Cartan gap block; the harmonic metric is obtained exactly
from the inverse coordinate map.

Every gap basis function vanishes at `ri=0`. The full wavefunction is recovered
by signed permutation and the chamber wavefunction is scaled by `sqrt(N!)`.
Consequently collision amplitudes and the relative antisymmetry residual are
exactly zero by construction. A separate `N=4`, six-point materialized audit
confirms both zeros; this materialization is a truth check, not the production
representation.

## Functional bases and operators

The center-of-mass site uses the full-line harmonic-oscillator basis at its
natural length `1/sqrt(N)`. Distance sites support two alternatives:

1. Dirichlet sine functions on `(0,Rmax)`, with independent basis order `D`
   and outer-box length `Rmax`.
2. Odd Hermite functions restricted to the positive half-line, with independent
   `D` and length scale and no outer box.

Overlap, derivative, kinetic, position, and position-square matrices have
analytic identities or independent quadrature tests. The projected
`-d^2/dx^2` matrix is formed before truncation; it is not replaced by the square
of a boundary-truncated first derivative.

For three noninteracting fermions, the exact energy is `4.5`. At fixed sine
box `Rmax=6`, orders `4,6,8,10` have errors
`8.53e-2,1.79e-2,6.40e-3,3.09e-3`. Odd Hermite with scale `0.7` gives errors
`1.17e-2,1.65e-3,6.09e-4` at orders `6,8,10`. This comparison is numerical
evidence that the unbounded candidate is more efficient here. It is not yet
available for the interacting MPO.

## Native Hamiltonian representation

The noninteracting Hamiltonian consists of exact one-site kinetic/trap terms,
adjacent mixed derivatives, and all relative-coordinate quadratic products.
The current direct-sum MPO has raw channel count

```text
W0 = (N^2 + 3N - 2) / 2.
```

For a pair `i<j`, the separation is a sum of consecutive positive gaps. A
degree-`K` Chebyshev polynomial on the corresponding finite interval is
propagated by a `K+1`-state power automaton. Direct summation over all pairs has
conservative raw bond `O(N^2 K)`, independent of local basis order. Projected
local powers are computed before MPO assembly, avoiding powers of a truncated
position matrix.

Production energy and AD use latticeTN's native MPS/MPO contractions. They do
not gather the `D**N` coefficient tensor. Product-basis vectors appear only in
bounded post-training Lanczos and TT-SVD truth audits, where the squared dense
Hamiltonian is still never constructed.

## Independent error controls

### N=2 basis and box

The independent continuum reference `2.553831733979` is a second-order
Richardson extrapolation of a relative-coordinate half-line finite-difference
solver.

| `D` at `Rmax=9` | Ground energy | Error vs reference |
|---:|---:|---:|
| 6 | `2.5540625227` | `2.308e-4` |
| 8 | `2.5539082157` | `7.648e-5` |
| 10 | `2.5538394782` | `7.744e-6` |
| 12 | `2.5538326731` | `9.391e-7` |
| 16 | `2.5538317365` | `2.521e-9` |

At fixed `D=20`, boxes `Rmax=5,6,7,8,9,10` have errors
`5.884e-4,5.100e-6,-5.290e-9,-1.960e-8,-1.902e-8,-1.570e-8`. The plateau from
seven onward separates the outer wall from finite basis resolution.

### Interaction separation and quadrature

At `(N,D,Rmax)=(2,12,9)`, degrees `K=8,12,16,20,24,32` differ from the `K=32`
energy by `2.874e-4,1.175e-5,2.410e-8,2.363e-9,6.855e-12,0`. Independent
sampled scalar errors decrease from `6.55e-3` to `6.21e-8`; they are diagnostics,
not interval certificates. Projection quadrature orders `48` through `220`
change the `K=20` energy by at most `1.73e-14`.

At N=4 and `(D,Rmax)=(10,4.5)`, degrees `8,12,16,20,24` give energies
`11.0209264,11.0273298,11.0273995,11.0274291,11.0274283`. The production
`K=20` point differs from `K=24` by `8.01e-7`; the low `K=8` value illustrates
why an apparently favorable total energy can be a nonvariational operator
approximation.

### N=4 basis and scale

The comparison reference `11.0230828537` is an independent exterior
harmonic-basis result at `D=14`. It is a numerical reference, not a continuum
bound.

| Sine `D`, fixed `Rmax=4.5` | Galerkin energy | Error vs exterior reference |
|---:|---:|---:|
| 6 | `11.0457787894` | `2.270e-2` |
| 8 | `11.0317438510` | `8.661e-3` |
| 10 | `11.0274291400` | `4.346e-3` |
| 12 | `11.0256076224` | `2.525e-3` |

At fixed `D=10`, `Rmax=3.5,4.0,4.5,5.0` gives
`11.0413285,11.0274422,11.0274291,11.0291167`. The two-sided deterioration
identifies both the small-box wall and the loss of resolution in an oversized
box.

### MPS bond

The independent `(D,Rmax,K)=(10,4.5,20)` Galerkin ground state was compressed
by standard TT-SVD and evaluated with the native MPO.

| Maximum bond | Energy error | Fidelity |
|---:|---:|---:|
| 1 | `4.016e-1` | `0.9409182` |
| 2 | `2.102e-2` | `0.9985693` |
| 4 | `1.472e-5` | `0.999999575` |
| 8 | `3.445e-9` | `0.999999999922` |
| 16 | `<3e-13` | `1-1.1e-16` |

The production bond 32 is therefore far above the observed representation
requirement. This table is a capacity audit, not a claim that low-bond random
optimization is equally easy.

## Blind AD and device parity

All truth values used for pass/fail are read or constructed after the formal
blind runs. N=2 uses `(D,Rmax,K,chi)=(12,9,20,12)` and a declared `1e-5`
Galerkin tolerance.

| Seed | Final energy | Error vs post-run Galerkin truth |
|---:|---:|---:|
| 1601 | `2.5538354021` | `2.727e-6` |
| 1602 | `2.5538356824` | `3.007e-6` |
| 1603 | `2.5538355139` | `2.838e-6` |

N=4 uses `(D,Rmax,K,chi)=(10,4.5,20,32)` and a declared `6e-3` total tolerance
against the exterior numerical reference.

| Seed | Final energy | Solver error vs Galerkin | Total error vs exterior reference |
|---:|---:|---:|---:|
| 1680 | `11.0274669704` | `3.783e-5` | `4.384e-3` |
| 1681 | `11.0274575099` | `2.837e-5` | `4.375e-3` |
| 1682 | `11.0274754819` | `4.634e-5` | `4.393e-3` |

Every final MPS has physical norm one. The N=4 MPS has 6,600 parameters, the
raw Hamiltonian MPO has maximum bond 59, and peak allocated GPU memory is below
33 MB. On an interacting N=3 parity state, CPU and RTX PRO 4000 Blackwell
energies agree exactly; the maximum gradient difference is `8.88e-15`.

## Error budget at the N=4 production point

| Source | Measured scale |
|---|---:|
| Sine basis plus outer box vs exterior `D=14` reference | `4.346e-3` |
| Interaction degree `K=20` vs `K=24` | `8.01e-7` |
| MPS representation, bond 32 | below `3e-13` audit resolution |
| Blind optimization | `2.84e-5`--`4.63e-5` |
| Exterior reference continuum remainder | unknown; reference is not a bound |

The dominant observed discrepancy is the finite sine basis, not the MPS bond,
interaction polynomial, or optimizer. The final total error remains below the
predeclared `6e-3` threshold for every seed.

## Prior-art and naming boundary

Hong et al. supply the orthonormal functional basis, operator projection,
coefficient MPS, and global-AD parent. Li--Waintal supply the direct ordered
first-quantized distance-MPS route, including small MPOs, DMRG/TDVP, and
controlled distance cutoffs. Phase 16 combines these ingredients in a
continuous COM/half-line implementation and closes its error axes.

No priority claim is made for ordered sectors, distance coordinates, or
first-quantized fermionic MPS. The method is called an ordered-distance
functional tensor network, not FEMPS. Generic matrix-wedge FEMPS remains
conditionally obstructed and the full-space exterior carrier remains a
separate mathematical program.

## Gate D decision and remaining limits

Gate D passes because:

1. the coordinate map, signed fermion recovery, collision wall, and
   normalization are exact;
2. functional operator matrices pass analytic or independent quadrature tests;
3. norm, Hamiltonian, and AD use native polynomial MPS/MPO contraction;
4. basis/scale, interaction degree, quadrature, MPS bond, and optimization
   errors are independently controlled; and
5. all blind N=2 and N=4 seeds meet their declared truth tolerances.

This does not establish an unbounded interacting basis, compressed-MPO safety,
`N>4` accuracy, or a favorable asymptotic accuracy-to-rank law. Those questions
define the next gate.

## Reproduction

```powershell
.\.venv\Scripts\python scripts\benchmark_ordered_continuous_controls.py
.\.venv\Scripts\python scripts\benchmark_ordered_continuous_training.py --device auto
```

Machine-readable records are
`results/phase16_ordered_continuous_controls.json` and
`results/phase16_ordered_continuous_training.json`.
