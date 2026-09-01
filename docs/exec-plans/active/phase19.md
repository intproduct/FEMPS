# Active execution plan: Phase 19 local-solver resources and N=8 refinement

## Objective

Close the two qualifications left by Gate F: reduce the N=8 MPO-bond gradient
sensitivity under a physically meaningful tangent-space audit, and make the
independent local optimizer resource-safe at chi 32. Strengthen N=8 basis and
reference convergence before considering any larger-particle point.

## Checkpoints

- [ ] Profile the N=8,D=10 two-site effective-Hamiltonian contraction and
  identify the einsum path responsible for the 78.12 GiB intermediate.
- [ ] Implement a matrix-free contraction ordering whose peak memory is
  bounded by explicit environment/MPO/MPS dimensions, with dense small-system
  and existing DMRG regression tests.
- [ ] Re-run the independent N=8 chi-32 local optimizer under a predeclared
  time/memory/iteration budget; retain chi 16 as the fallback control.
- [ ] Replace or complement raw tensor-parameter gradient comparison by a
  gauge-fixed/tangent-space directional derivative audit, without deleting the
  Gate F auxiliary miss.
- [ ] Compare training MPO bonds 128, 160, and 192 under matched seeds and
  schedules; predeclare energy and tangent-gradient tolerances before choosing
  a new production bond.
- [ ] Extend multiscale basis controls to D=12 where conditioning permits, and
  add Fourier/local-quadrature convergence at the selected N=8 basis.
- [ ] Strengthen the N=8 external reference or construct a bounded alternative
  error bracket; continue to label current exterior values numerical.
- [ ] Reassess N=2,4,6,8 accuracy per stored MPO entry, peak memory, and wall
  time before any N=10 admission decision.
- [ ] Issue Gate G on local-solver resource closure and N=8 refinement.

## Exit criterion

Gate G passes only if chi-32 local optimization no longer exceeds the RTX PRO
4000 memory budget, the chosen production MPO bond passes both energy and a
gauge-aware derivative audit, and the refined N=8 basis/operator/reference
budget remains controlled. Failure retains Gate F at its qualified N=8 scope
and does not admit N=10.

