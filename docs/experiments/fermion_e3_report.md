# E3 report: four noninteracting spinless fermions

## Outcome

Four spin-polarized fermions in the one-dimensional unit-frequency harmonic
trap occupy functional orbitals `n=0,1,2,3`, so

\[
 E_0=\sum_{n=0}^3(n+1/2)=8.
\]

At basis order `D=8`, the direct Slater construction, a correlation-bond-one
matrix-wedge FEMPS, a two-pair-channel Pfaffian embedding, the polynomial AGP
energy, and an independent `binom(8,4)=70` dimensional Slater--Condon
diagonalization all agree. The coefficient differences are at most `5.56e-17`,
the energy is exactly `8` in float64 arithmetic, and the stationary AD gradient
is zero.

The two pair channels are a structured Pfaffian parameterization of the same
four-form; they are not the reduced FEMPS correlation bond. The direct
decomposable four-form has internal correlation bonds `(1,1,1)`.

## Representation-complexity separation

The normalized Slater tensor has ordinary particle-TT ranks

\[
 (1,4,6,4,1),
\]

including boundary ranks. Its nonzero Schmidt values are flat:

| Particle cut | Multiplicity | Singular value | Maximum error |
|---:|---:|---:|---:|
| 1 | 4 | 1/2 | 0 |
| 2 | 6 | 1/sqrt(6) | 1.11e-16 |
| 3 | 4 | 1/2 | 0 |

Therefore the central-cut best ordinary rank-`r` relative error is
`sqrt((6-r)/6)`. The measured errors agree to roundoff:

| Retained rank `r` | Best relative error |
|---:|---:|
| 1 | 0.912871 |
| 2 | 0.816497 |
| 3 | 0.707107 |
| 4 | 0.577350 |
| 5 | 0.408248 |
| 6 | 0 |

This is the planned clean separation between analytically carried exchange
multiplicity and variational correlation complexity. It is a theorem-backed
Slater example, not evidence that every interacting four-form remains at
correlation bond one.

## Blind AD reproduction

A dense complex Pfaffian pair matrix was initialized without the exact
orbitals and optimized through the polynomial functional-basis energy on RTX
PRO 4000 Blackwell:

| Quantity | Value |
|---|---:|
| Initial energy | 16.441557530195 |
| Final/best energy | 7.999999999999981 |
| Absolute error | 1.87e-14 |
| Ground-state fidelity | 1 within roundoff |
| Steps | 600 |
| Wall time | 7.21 s |
| Peak CUDA allocation | 17,074,176 bytes |

The best recorded state is restored; the terminal iterate was
`8.000000000000014`. Raw coefficients are never materialized during
optimization. The small explicit tensor is used only after training for the
rank and antisymmetry diagnostics.

Raw evidence is in `results/fermion_e3.json`.

## Next benchmark

E4 adds the all-to-all harmonic interaction and scans interaction strength,
functional basis order, single-AGP error, and finite-AGP-sum improvement against
the exact exterior-sector Hamiltonian.
