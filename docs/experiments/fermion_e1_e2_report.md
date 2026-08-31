# Phase 4 report: first functional Pfaffian fermion solver

## Outcome

The conditionally admitted fixed-number Pfaffian state is now connected to the
2201 harmonic functional basis, one-body operator matrices, and factorized
two-body interactions. Deterministic AD optimization on RTX PRO 4000 Blackwell
reaches both E1 and E2 full-reference ground energies.

| Quantity | E1 noninteracting | E2 interacting |
|---|---:|---:|
| Basis order `D` | 12 | 12 |
| Interaction `kappa` | 0 | 0.35 |
| Initial energy | 11.1617192247 | 13.7141150636 |
| Final energy | 2.000000000000 | 2.455760721565 |
| Finite-basis exact energy | 2.000000000000 | 2.455760721565 |
| Error vs finite-basis truth | 6.66e-16 | 8.88e-16 |
| Continuum exact energy | 2.000000000000 | 2.455760721561 |
| Error vs continuum | 0 | 4.58e-12 |
| Pair decomposition length | 1 | 6 |
| Ordinary particle TT rank | 2 | 12 |
| Antisymmetry residual | 0 | 0 |
| GPU wall time, 1000 steps | 25.21 s | 25.27 s |

The E2 continuum formula follows separation into a center-of-mass oscillator
of frequency `omega` and an odd relative oscillator of frequency
`sqrt(omega^2+2*kappa)`:

\[
 E_{\rm E2}=\frac{\omega}{2}
 +\frac{3}{2}\sqrt{\omega^2+2\kappa}.
\]

## Functional operator representation

The interaction

\[
 \frac{\kappa}{2}(x_1-x_2)^2
 =\frac{\kappa}{2}(x^2\otimes I+I\otimes x^2)
 -\kappa x\otimes x
\]

has operator-Schmidt rank two. `x`, the exact projected `x^2`, and the harmonic
one-body Hamiltonian are all functional-basis matrices. No coordinate grid or
occupation-number MPS is introduced.

## Validation

- The polynomial Pfaffian energy equals an explicit `D^2` particle tensor for
  arbitrary random pair matrices.
- The finite-basis truth is obtained by independent diagonalization in the
  normalized increasing two-form basis.
- The `D=4,6,8,10,12,14` finite-basis E2 errors decrease monotonically from
  `1.01e-3` to `2.93e-14`.
- A 12-step interruption followed by checkpoint resume gives the same final
  result as uninterrupted 30-step training within `2e-13`.
- Every optimizer projection occurs after the Adam step and outside the loss
  graph, following the latticeTN Rayleigh-optimization convention.

Raw records and resumable local checkpoints are under
`docs/experiments/results/fermion_e1_e2/`; checkpoint binaries are ignored by
Git.

## Basis and constrained-pair scan

The independent antisymmetric diagonalization gives the following convergence
to the continuum E2 energy.

| Basis order `D` | Finite-basis energy | Error vs continuum |
|---:|---:|---:|
| 4 | 2.456770326268 | 1.010e-3 |
| 6 | 2.455771253450 | 1.053e-5 |
| 8 | 2.455760810490 | 8.893e-8 |
| 10 | 2.455760722225 | 6.638e-10 |
| 12 | 2.455760721565 | 4.576e-12 |
| 14 | 2.455760721561 | 2.931e-14 |

At fixed `D=12`, constrain the pair matrix to

\[
 F=\sum_{a=1}^{\chi_{m pair}}w_a
 (u_av_a^{\mathsf T}-v_au_a^{\mathsf T}).
\]

This scan is a representation diagnostic: each run starts from the truncated
real skew-canonical decomposition of the independently known ground-state pair
matrix and then applies AD refinement. It is not evidence that blind nonlinear
optimization finds the same solution from a random initialization.

| Pair channels | Error vs finite-basis truth | Error vs continuum |
|---:|---:|---:|
| 1 | 2.698e-4 | 2.698e-4 |
| 2 | 1.767e-8 | 1.768e-8 |
| 3 | 7.29e-13 | 5.305e-12 |
| 4 | < 1e-15 | 4.577e-12 |
| 5 | < 1e-15 | 4.577e-12 |
| 6 | < 1e-15 | 4.577e-12 |

The optimizer restores the best recorded state. For channels three through six,
the terminal Adam iterate drifted above the canonical initialization by between
`2.6e-9` and `1.7e-8` in continuum energy. Both values remain in the raw JSON;
this exposes the nonconvex factor gauge instead of hiding it behind the restored
answer. The real skew decomposition is only an oracle initializer here; a
complex skew-Takagi initializer remains future work.

## Interpretation

E1 has one decomposable pair and realizes the required correlation-bond-one
sanity condition. The unconstrained E2 solution has algebraic skew-pair length
six in the selected functional basis, but three canonical pair channels already
reach `7.3e-13` representation error. Exchange alone still gives ordinary
particle rank two, while the interacting solution reaches rank twelve. This is
the first numerical example in the repository where the exterior/Pfaffian
correlation control and ordinary particle-cut rank visibly separate.

These two-particle tests validate the functional operator and exterior
optimization path, but do not establish a four-fermion representation
advantage. That claim is reserved for a later finite-AGP-sum benchmark.
