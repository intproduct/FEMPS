# Blocked Pfaffian report: odd particle number

## Outcome

The conditionally admitted functional Pfaffian family now covers odd particle
number through

\[
 \Psi_M^{\rm block}(F,u)=u\wedge\Omega_F^{\wedge M}/M!.
\]

The implementation provides ordered matrix-wedge FEMPS cores, exterior
coefficients, an exponential particle-tensor truth oracle, and polynomial norm,
overlap, one-body, and factorized two-body contractions. It reuses the even AGP
engine by adding one auxiliary functional direction and subtracting the
auxiliary-unoccupied sector.

## Independent small-system truth

Deterministic complex128 validation at `D=5,N=3` gives:

| Comparison | Maximum/absolute error |
|---|---:|
| Ordered FEMPS vs Pfaffian coefficients | 2.31e-16 |
| Polynomial vs explicit norm | 4.44e-16 |
| Polynomial vs explicit transition overlap | 4.58e-16 |
| Polynomial vs particle-tensor one-body value | 7.37e-15 |
| Polynomial vs particle-tensor two-body value | 4.95e-16 |
| Polynomial vs explicit norm gradient | 1.13e-14 |

The `N=1` limit is separately tested: pair-matrix dependence cancels exactly,
leaving the ordinary blocked-orbital norm and one-body matrix element.

## Blackwell parity

For a dense random `D=32,N=21` state and Hermitian functional one-body matrix:

| Quantity | CPU/RTX PRO 4000 Blackwell difference |
|---|---:|
| Normalized energy | 5.07e-11 |
| Pair-matrix gradient, maximum absolute | 4.56e-12 |
| Blocked-orbital gradient, maximum absolute | 5.80e-12 |

The GPU is `cuda:2`, compute capability 12.0. The two V100 devices are skipped
because the installed CUDA 12.8 PyTorch wheel does not contain their target
architecture. Raw evidence is in `results/blocked_agp_validation.json`.

## Scope

This closes the single-block odd-particle engineering item. It does not yet
cover several blocked orbitals, spin adaptation, or prove a representation
advantage over ordered-sector alternatives. The next physics milestone remains
an interacting four-fermion finite-AGP benchmark.
