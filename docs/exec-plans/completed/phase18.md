# Completed execution plan: Phase 18 basis efficiency and N=8 gate

## Objective

Reduce the N=6 half-line basis error exposed by Gate E and determine whether
the unbounded ordered-distance functional TN admits a controlled N=8 point
without temporary dense raw-MPO storage becoming the dominant resource.

## Checkpoints

- [x] Compared a collision-compatible two-scale odd-Hermite basis against the
  single-scale basis at matched N=2 and N=4 orders, with analytic overlap/
  kinetic matrices and independent quadrature.
- [x] Extended N=6 to D=10 matrix-free Galerkin truth and reduced the Gate E
  basis error from `1.328e-2` to `2.452e-3`.
- [x] Replaced dense raw Fourier-MPO storage by an incremental sparse-
  recurrence/left-SVD builder and matched raw-then-compress globally at
  N=3,4,5.
- [x] Audited N=6,D=10 MPO bonds 128/192 on a one-million-dimensional global
  action and N=8,D=10 bonds 128/192 on a fixed-state energy and full gradient.
- [x] Added predeclared staged global AD and separated capacity/optimizer error
  with a right-canonical chi-16 local DMRG audit.
- [x] Recorded the chi-32 local-DMRG 78.12 GiB resource rejection instead of
  hiding it.
- [x] Completed a blind N=8,D=10 Blackwell point without product-basis gather,
  recording all basis/operator/MPO/MPS/optimization/time/memory controls.
- [x] Reassessed the controlled N=2,4,6,8 accuracy/resource trend and retained
  the numerical-reference and non-asymptotic qualifications.
- [x] Issued ADR 0008 and Gate F.

## Gate F result

Gate F is **PASS (controlled N=8 point, qualified auxiliary audit)**. The N=6
D=10 basis error is `2.452e-3`, an `81.5%` reduction from Gate E. The N=8 best
blind error against an exterior D=12 numerical reference is `8.364e-3`, below
the declared `1.2e-2` budget, and the independent local optimizer differs by
only `5.11e-5`.

The production builder never materializes dense raw Fourier bulk tensors. The
N=8,D=10 maximum build intermediate is 5,465,600 entries versus 109,482,800
theoretical raw entries.

The additional bond-128/192 raw parameter-gradient threshold misses
(`2.90e-5` versus `2e-6`) even though fixed-state energy convergence passes
and gradient cosine similarity is `0.999999999584`. Chi-32 local DMRG is also
resource rejected. Both qualifications carry into Phase 19. No N=10 or
asymptotic claim is made.

