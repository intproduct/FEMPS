# Completed execution plan: Phase 19 local-solver resources and N=8 refinement

## Objective

Close the two qualifications left by Gate F: reduce the N=8 MPO-bond gradient
sensitivity under a physically meaningful tangent-space audit, and make the
independent local optimizer resource-safe at chi 32. Strengthen N=8 basis and
reference convergence before considering any larger-particle point.

## Checkpoints

- [x] Profiled the N=8,D=10 two-site effective-Hamiltonian contraction and
  identified the einsum path responsible for the 78.12 GiB intermediate.
- [x] Implemented a matrix-free contraction ordering bounded by explicit
  environment/MPO/MPS dimensions, with dense and DMRG regression tests.
- [x] Re-ran the independent N=8 chi-32 local optimizer under the predeclared
  2 GiB budget; peak CUDA memory is 736,884,736 bytes.
- [x] Complemented the Gate F raw tensor-gradient miss with a left-gauge,
  many-body-normalized physical-tangent derivative audit.
- [x] Compared training MPO bonds 128, 160, and 192 under matched seeds and
  schedules; bond 128 is the smallest passing production value.
- [x] Extended the multiscale basis to a blind D=12 point and closed Fourier,
  local-quadrature, basis-conditioning, memory, and optimizer controls.
- [x] Extended the exterior N=8 numerical reference from D=12 to D=14 with a
  Q128/Q160 quadrature difference of `9.24e-13`.
- [x] Reassessed N=2,4,6,8 accuracy, stored MPO entries, memory, and time;
  retained the non-asymptotic qualifications and deferred N=10.
- [x] Issued ADR 0009 and Gate G.

## Gate G result

Gate G is **PASS (resource-closed controlled N=8 point)**. The staged chi-32
local action stays below 2 GiB, bond 128 passes the gauge-aware tangent audit,
and the blind N=8,D=12 production point has error `7.174e-3` against the
exterior D14 Q160 numerical reference. This is `17.6%` smaller than Gate F's
D10 error against the same reference.

The historical Gate F raw-parameter gradient miss remains recorded. D14 is not
a continuum bound, the N=2/4/6/8 comparison is descriptive rather than an
asymptotic fit, and N=10 is not admitted.
