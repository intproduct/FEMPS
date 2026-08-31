# E4 report: four interacting harmonic fermions

## Model and exact reference

The benchmark Hamiltonian is

\[
 H=\sum_{i=1}^4\left(-\frac12\partial_{x_i}^2+\frac12x_i^2\right)
 +\frac{\kappa}{2}\sum_{i<j}(x_i-x_j)^2.
\]

Center-of-mass separation and the fermionic Vandermonde degree give

\[
 E_{\rm cont}=\frac12+\frac{15}{2}\sqrt{1+4\kappa}.
\]

An independent Slater--Condon Hamiltonian in `Lambda^4 V_D` supplies every
finite-basis truth. It reduces to the earlier explicit two-particle projection
in regression tests and is never used by the production Pfaffian contraction.

## Basis convergence

Absolute errors of the finite exterior-sector truth against the continuum are:

| `D` | `kappa=0.1` | `kappa=0.35` | `kappa=0.8` |
|---:|---:|---:|---:|
| 4 | 1.259e-1 | 1.131e0 | 4.130e0 |
| 6 | 3.008e-3 | 1.409e-1 | 1.003e0 |
| 8 | 4.360e-5 | 1.372e-2 | 2.334e-1 |
| 10 | 4.671e-7 | 1.024e-3 | 4.670e-2 |
| 12 | 4.161e-9 | 6.313e-5 | 7.955e-3 |

This table prevents continuum truncation from being folded into an ansatz
error. Stronger coupling requires a substantially larger oscillator basis.

## Blind single-AGP scan

At `D=8`, three independent dense complex initializations were optimized for
600 steps on RTX PRO 4000 Blackwell.

| `kappa` | Best error vs finite truth | Mean error | Seed spread |
|---:|---:|---:|---:|
| 0.1 | 4.772e-5 | 4.869e-5 | 2.132e-6 |
| 0.35 | 2.804e-3 | 2.963e-3 | 3.437e-4 |
| 0.8 | 7.955e-3 | 1.144e-2 | 1.016e-2 |

All values are variational upper bounds. Polynomial energies agree with an
independent explicit exterior-vector expectation within `1.6e-13`.
Optimization becomes visibly nonconvex at strong coupling.

For `kappa=0.35` and fixed seed zero, the basis/ansatz split is:

| `D` | Basis error | Single-AGP error vs finite truth | Total continuum error |
|---:|---:|---:|---:|
| 6 | 1.409e-1 | 4.065e-3 | 1.449e-1 |
| 8 | 1.372e-2 | 3.148e-3 | 1.687e-2 |
| 10 | 1.024e-3 | 2.510e-3 | 3.534e-3 |
| 12 | 6.313e-5 | 2.218e-3 | 2.282e-3 |

By `D=12`, the result is clearly ansatz-limited rather than basis-limited.

Every converged interacting single AGP at `D=8` has full exterior support
`70/70` and ordinary particle-TT internal ranks `(8,28,8)`. The noninteracting
Slater ranks were `(4,6,4)`. Ordinary particle-TT therefore absorbs both the
exchange multiplicity and correlation growth, while the Pfaffian description
continues to use one skew pair matrix.

## Finite-AGP representation hierarchy

At `D=8,kappa=0.35`, the exact finite-basis eigenvector was used to fit the
span of `K` AGPs. This is explicitly an oracle representation diagnostic and
initializer, not a blind solver. The best of three fitting seeds gives:

| `K` | Infidelity | Error vs finite truth | Polynomial/exterior difference | Polynomial evaluation |
|---:|---:|---:|---:|---:|
| 1 | 4.074e-4 | 3.106e-3 | 3.55e-15 | 0.012 s |
| 2 | 8.922e-7 | 1.107e-5 | 5.33e-15 | 0.042 s |
| 4 | 2.800e-8 | 4.988e-7 | 1.60e-14 | 0.166 s |
| 8 | 1.120e-10 | 2.813e-9 | 1.78e-14 | 0.653 s |

The observed evaluation time follows the expected `K^2` transition count.
The hierarchy establishes that modest finite sums represent this interacting
four-fermion target to high precision; it does not make `K` a proven canonical
FEMPS spectrum or entropy.

Starting from the oracle `K=2` state, 200 steps of production polynomial-energy
AD retain a `1.040e-5` variational error. The same optimizer from three random
`K=2` initializations ends at errors `4.261e-3`, `2.101e-3`, and `3.326e-3`.
Thus representation capacity passes, while blind nonlinear optimization is now
the dominant failure mode. The oracle is retained only as a reproducible
initializer for developing a better alternating/generalized-eigenvalue solver.

## Decision

Continue the restricted Pfaffian/finite-AGP route. It passes the E4
representation test by more than six orders of magnitude from `K=1` to `K=8`,
so the ordered-sector fallback is not triggered on capacity grounds. Phase 6
must improve gauge handling and blind optimization before claiming a practical
many-fermion solver.

Raw evidence:

- `results/fermion_e4_truth_sweep.json`;
- `results/fermion_e4_single_agp_sweep.json` and the `d6/d10/d12` companion
  records;
- `results/fermion_e4_agp_rank_sweep.json`;
- `results/fermion_e4_k2_energy_refinement.json`.
