# Completed execution plan: Phase 17 unbounded interaction and scaling gate

## Objective

Remove the finite sine-box bottleneck identified by Gate D, build an
interaction representation compatible with an unbounded half-line basis, and
establish a controlled larger-particle point with globally audited MPO
compression.

## Checkpoints

- [x] Derived and tested the exact Fourier--Bessel cosine separation for soft
  Coulomb with explicit frequency cutoff and quadrature order.
- [x] Implemented odd-Hermite cosine/sine projection matrices and checked
  one- and two-distance operators against independent half-line quadrature.
- [x] Compared sine and odd-Hermite bases at matched N=2 and N=4 truth points,
  with independent basis-order and scale controls.
- [x] Replaced direct all-pair summation by a four-real-state recurrence per
  Fourier node, giving interaction bond `4M` independent of particle count.
- [x] Audited compact/direct equivalence and every adopted compression with a
  small dense operator or bounded global action error.
- [x] Separated optimizer and MPS capacity by post-training Lanczos and TT-SVD.
- [x] Completed blind N=6,D=8 interacting GPU training without a product-basis
  gather, recording time, peak memory, raw/compressed MPO resources, MPS bond,
  norms, gradients, scale, and all approximation controls.
- [x] Revisited the Li--Waintal/Hong boundary and retained the narrow
  integration/evidence claim and non-FEMPS naming.
- [x] Issued ADR 0007 and Gate E.

## Gate E result

Gate E is **PASS (controlled unbounded N=6 prototype)**. At matched N=2 and
N=4 basis order, interacting odd Hermite improves the finite sine box. The
compact all-pair MPO agrees with the direct-pair construction to float64
precision, and the N=6 training compression has `1.63e-9` relative global
action error.

Three blind N=6 seeds finish `1.29e-4--2.48e-4` above their post-run same-basis
Galerkin truth with fidelity above `0.99998`. The Galerkin energy differs from
an exterior D=12 numerical reference by `1.328e-2`, below the declared `2e-2`
controlled-point tolerance. TT-SVD shows MPS bond eight has `4.50e-6` capacity
error, identifying the local functional basis as the dominant next bottleneck.

The result does not establish N=8 accuracy, a continuum convergence rate, or
an asymptotic resource advantage.
