# Phase 8 competing-representation and N=8 report

## E8a noninteracting polynomial baseline

The first Phase 8 point uses eight spin-polarized harmonic fermions in a
`D=10` oscillator basis. The occupied orbitals `n=0,...,7` give

\[
E_0=\frac{N^2}{2}=32.
\]

The constructed four-channel Pfaffian, polynomial contraction, and independent
45-dimensional exterior-sector Hamiltonian agree exactly in float64 arithmetic.
The ordinary `D^N` particle tensor has `10^8` entries and is never constructed.

The Slater theorem predicts ordinary particle-TT ranks

\[
(1,8,28,56,70,56,28,8,1),
\]

whereas all seven FEMPS correlation bonds are one. Even ordinary central rank
69 leaves best relative error `1/sqrt(70) = 0.11952...`.

A blind `K=1` polynomial AD run on RTX PRO 4000 Blackwell gives:

| Quantity | Value |
|---|---:|
| Initial energy | 40.008200626801 |
| Final polynomial energy | 31.99999999999972 |
| Independent exterior energy | 32.00000000000000 |
| Polynomial/exterior difference | 2.81e-13 |
| Ground-state fidelity | 1 within roundoff |
| Steps | 600 |
| GPU optimization time | 15.86 s |
| Peak CUDA allocation | 18,343,936 bytes |
| Ordinary particle tensor materialized | no |
| Exterior truth dimension | 45 |

This is a measured finite point, not an asymptotic scaling claim. It verifies
that the existing polynomial Pfaffian path reaches eight particles without
ordinary-tensor materialization. The ordered-sector comparison and interacting
`N=8` point remain open.

Raw evidence is in `results/fermion_e8a.json`.

## Ordered-sector harmonic-grid oracle

The comparator is defined by the normalized isometry

\[
\bar\Psi=\sqrt{N!}\,\Psi\big|_{x_1<\cdots<x_N}
\]

with a hard Dirichlet wall on particle coincidences. Its inverse sorts the
coordinates, restores the permutation sign, and divides by `sqrt(N!)`.
Explicit `D=5,N=3` tensor tests preserve the norm and reconstruct the complete
antisymmetric tensor exactly within float64 roundoff.

For a local finite-difference harmonic Hamiltonian, the direct ordered-grid
oracle was compared with the independent exterior Slater--Condon lift. At
`N=3`, grid sizes `7,9,11,13` give ordered dimensions `35,84,165,286` and zero
matrix difference at all four points. The ordered ground energy also agrees
with the sum of the three lowest one-particle grid energies to at most
`4.36e-14`.

The finest grid in this initial locality test has spacing `0.6`, box
`[-3.6,3.6]`, and energy `4.272746883511`, versus continuum `4.5`. The
`-0.2273` difference is intentionally recorded as coarse finite-difference and
finite-box error; it is not ordered-sector ansatz error. A controlled grid or
functional-basis comparison is still required before performance claims.

Raw evidence is in `results/ordered_sector_harmonic.json`; definitions and the
three-way comparison protocol are in `docs/theory/ordered_sector.md`.

## E8b interacting harmonic point

At `N=8,D=10,kappa=0.02`, a single AGP was continued from the E8a
noninteracting state without using exact-state information. The analytic
continuum energy, finite exterior truth, and variational result are:

| Quantity | Value |
|---|---:|
| Continuum energy | 34.426538284947 |
| Finite-basis truth | 34.427993016199 |
| K=1 variational energy | 34.427996725375 |
| Basis error | 1.455e-3 |
| Ansatz/optimizer error vs finite truth | 3.709e-6 |
| Polynomial/exterior difference | 7.82e-14 |
| Finite-basis ground fidelity | 0.9999991585 |
| GPU optimization time | 63.72 s |
| Peak CUDA allocation | 19,486,208 bytes |

Only the 45-dimensional exterior truth sector is materialized; the `10^8`
ordinary particle coefficients are not. The reported `K=1` error combines any
remaining single-AGP representation error and finite optimization error.

Raw evidence is in `results/fermion_e8b_single_agp.json`.

## Gate B decision

Phase 8 supports complementary, not interchangeable, roles:

- finite-AGP FEMPS remains the production route into E6 because it already
  uses the 2201 orthonormal functional basis, exact exterior antisymmetry, AD,
  and polynomial one-/two-body contractions through `N=8`;
- the ordered sector is an exact competing representation and removes exchange
  multiplicity, but the current implementation is only a small local-grid
  oracle. It is retained as an independent boundary/sign control until a
  controlled functional or grid MPS solver is implemented;
- ordinary particle TT remains a no-go control and is materialized only where
  safe.

This is a readiness decision, not a claim that FEMPS is asymptotically better
than ordered-sector MPS. E6 should use finite-AGP FEMPS as the main solver and
ordered-sector calculations as small-system complementary controls.
