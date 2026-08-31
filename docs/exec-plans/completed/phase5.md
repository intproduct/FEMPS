# Completed execution plan: Phase 5 four-fermion benchmarks

## Objective

Establish the first four-particle representation-complexity separation and then
measure how the admitted Pfaffian/finite-AGP hierarchy behaves under an exactly
solvable symmetric harmonic interaction.

## Completed checkpoints

- [x] Add an independent general-`N` Slater--Condon exterior-sector truth
  Hamiltonian and reduce it to the earlier two-particle oracle.
- [x] E3: construct four noninteracting fermions as both correlation-bond-one
  FEMPS and a two-pair-channel Pfaffian state.
- [x] E3: verify energy, flat particle Schmidt spectra, ordinary TT ranks
  `(1,4,6,4,1)`, truncation errors, antisymmetry, and blind Blackwell AD.
- [x] E4: scan interaction strength and functional basis order against the
  analytic continuum energy and finite-basis exterior truth.
- [x] Compare one AGP with finite sums of `K` AGPs using independent truth,
  variational margins, and repeat-seed optimization stability.
- [x] Measure ordinary particle-TT ranks and support diagnostics only where
  explicit post-training materialization remains safe.
- [x] Decide whether modest `K` establishes a useful interacting four-fermion
  representation advantage or triggers the ordered-sector fallback.

## Decision and exit result

The representation hierarchy passes: at `D=8,kappa=0.35`, oracle-fitted
`K=1,2,4,8` errors against finite truth are `3.11e-3`, `1.11e-5`, `4.99e-7`,
and `2.81e-9`. All are variational, continuum and basis errors are separated,
and polynomial values agree with exterior truth. Blind random `K=2`
optimization remains unreliable, so Phase 6 targets solver conditioning rather
than further ansatz expansion.
