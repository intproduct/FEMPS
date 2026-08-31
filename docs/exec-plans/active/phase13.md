# Active execution plan: Phase 13 beyond-LC-AGP FEMPS structure gate

## Objective

Identify a genuinely non-LC-AGP exterior functional tensor-network structure
whose correlation multiplicity is distinct from standard AGP-CI and whose norm
and one-/two-body matrix elements admit a proved polynomial contraction; or
produce a precise obstruction and pivot decision.

## Checkpoints

- [ ] Prove and document the exact relation between the current finite-AGP
  subclass, LC-AGP/AGP-CI literature, and its matrix-wedge FEMPS embedding.
- [ ] Define the smallest nontrivial matrix-wedge family not reducible to a
  polynomial-size LC-AGP sum at fixed N.
- [ ] Derive N=4, chi=2 symbolic norm and one-body contractions in at least two
  independent forms.
- [ ] Test whether compound/exterior transfer states close at polynomial width
  for that family, with an explicit complexity bound or counterexample.
- [ ] Define a gauge-invariant statistics-carrier/correlation-multiplicity
  object whose single-Slater value is one; do not call it entanglement.
- [ ] Implement only small exact materialization and gradient oracles until the
  contraction gate is resolved.
- [ ] Audit the proposed structure against symmetry-adapted TN, geminal/APG,
  nonorthogonal AGP-CI, and first-quantized ordered-sector methods.
- [ ] Issue a Gate B decision: proceed with a beyond-LC-AGP solver, restrict to
  another proved subclass, or pivot to no-go/ordered-sector theory.

## Exit criterion

A nontrivial beyond-LC-AGP family must have a precise state definition,
small-system exterior equivalence, AD-compatible polynomial contraction with an
explicit complexity bound, and a surviving novelty distinction. Otherwise the
phase must record the obstruction and pivot rather than expand benchmarks.
