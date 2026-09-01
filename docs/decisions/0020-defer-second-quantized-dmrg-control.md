# ADR 0020: Defer second-quantized DMRG control

- Status: accepted
- Date: 2026-09-01
- Scope: Phase 30 method consolidation

## Context

The user permits second-quantized DMRG as an explicitly named external
comparator, but forbids presenting it as FEMPS. The current admitted N4 and N6
soft-Coulomb benchmarks use `D=10`. Their exterior dimensions are both 210, so
direct complex128 diagonalization of the unfactorized quadrature Hamiltonian is
cheap, independently verified, and supplies energy, eigenvector, variance,
materialized antisymmetry, and ordinary particle-TT ranks.

A second-quantized DMRG calculation at the same finite basis would approximate
that already exact 210-dimensional reference. It would add implementation and
convergence choices without strengthening the energy truth or the
first-quantized symmetry audit. Running at a larger orbital or particle space
only to make DMRG useful would violate the current stop on simultaneous size
expansion and would remove the strongest independent CI controls.

## Decision

Defer DMRG in Phase 30. Use direct exterior CI as the primary energy/reference
comparator, and retain Slater, AGP, ordinary particle TT, and ordered-sector
methods under their own names where already admitted.

DMRG may be reconsidered only if all of the following hold:

1. the proposed finite-basis exterior dimension makes direct CI materially
   infeasible;
2. an independent error-control plan exists for both DMRG truncation and FEMPS;
3. the comparison answers a physics or complexity question not already
   answered by the exact-CI region;
4. an ADR fixes orbital ordering, symmetries, bond-dimension/discarded-weight
   convergence, time, memory, software provenance, and claim language before
   execution;
5. the result remains explicitly second quantized and is never called FEMPS.

## Consequences

- No comparator is added merely to increase method count.
- Current FEMPS accuracy claims continue to use the stronger exact reference.
- No inference is made about FEMPS versus DMRG performance outside the tested
  region.
