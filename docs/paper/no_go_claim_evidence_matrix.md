# No-go manuscript claim/evidence matrix

## Purpose

This table is an internal writing constraint. “Admissible wording” is the
strongest current claim supported by the repository; “forbidden wording” lists
shortcuts that would overstate scope or novelty.

| Claim | Evidence level | Direct prior art/dependency | Admissible wording | Forbidden wording |
|---|---|---|---|---|
| Ordinary particle-TT bond floor | proof draft plus exact tests | standard TT unfolding-rank identity | every nonzero alternating `N`-tensor has particle-cut rank at least `binom(N,k)` | all fermionic tensor networks require exponential bond |
| Slater approximate floor | proof draft plus exact tests | Schmidt decomposition and Eckart--Young | a normalized Slater has a flat particle-cut spectrum, so ordinary particle-TT needs a binomial fraction of the bond for fixed relative error | every correlated fermionic state has a flat spectrum |
| Exterior carrier strict antisymmetry | elementary proof plus materialization tests | exterior algebra | matrix-wedge states are alternating by construction | exterior form implies efficient contraction |
| Generic matrix-wedge exact contraction | exact reduction draft plus order-4 certificate | Chien et al. noncommutative determinant hardness | unrestricted exact squared-norm contraction is #P-hard under a polynomial-time Turing reduction | no FEMPS approximation or restricted family can be efficient |
| Bounded coefficient algebra | theorem draft; `T_2`/`Mat_2` exact certificates | Wedderburn--Malcev, polarization, existing LC-AGP | fixed largest simple block and radical depth imply a polynomial exact LC-AGP expansion | every bounded algebra gives a novel efficient FEMPS |
| One-counter/fixed-state growing memory | theorem draft plus all-boundary certificates | interpolation, Waring/Veronese, weighted automata | these admitted graded memories remain polynomial LC-AGP | every growing radical collapses |
| Sparse growing-width path | elementary proof plus exact M<=6 certificate | APG/APIG, Fischer decomposition, Valiant permanent | the generic fixed-bandwidth class contains a bandwidth-one APG instance with #P-hard exact squared norm | APG is new; APG has proved exponential minimal AGP rank; every sparse instance is hard |
| Approximate sparse-path norm | transfer proof plus exact rational conditioning controls | Jerrum--Sinclair--Vigoda; Meiburg; Gurvits/Aaronson--Hance | entrywise-nonnegative coefficients admit an FPRAS, but a generic relative squared-norm PRAS would imply a PRAS for real-PSD permanents and hence `RP=NP`; additive estimates need a certified norm lower bound to control energy | exact permanent hardness by itself rules out approximation; every signed instance has a sign problem; unbiased estimates certify the Rayleigh quotient |
| Finite LC-AGP implementation | exact contraction/AD tests and continuum benchmarks | Uemura et al.; Dutta et al.; Kawasaki et al. | validated baseline/control integrated with the 2201 functional layer | a new AGP-CI ansatz |
| Ordered continuous route | controlled Gates D--G | Hong et al.; Li--Waintal | a reproducible integration/control route with separated continuum errors | first ordered-coordinate or first first-quantized MPS method |

## Complexity language

- Say “exact squared norm” for `<Psi|Psi>` in theorem statements.
- Say “would imply `FP=#P`” or “conditional on the standard permanent-
  complexity assumption”; do not state an unconditional runtime lower bound.
- Phase 13 is a Turing reduction using boundary and interference queries.
- Phase 22 is a one-query exact-value reduction followed by polynomial-time
  exact arithmetic and a nonnegative integer square root.
- Phase 24 addresses relative squared-norm approximation on a real-PSD
  specialization. It does not rule out additive approximation, the entrywise-
  nonnegative FPRAS cone, or promised structured families.
- A norm/numerator confidence event certifies energy only when its denominator
  interval stays strictly positive; unbiasedness alone is insufficient.

## Novelty language

The defensible project-specific package is the connection among:

1. the 2201 particle-coordinate functional-basis setting;
2. the ordinary particle-TT exchange-rank obstruction;
3. the explicit FEMPS/matrix-pair reduction hierarchy; and
4. the controlled ordered-coordinate fallback evidence.

The ingredients TT ranks, exterior algebra, AGP/APG, Fischer decompositions,
permanents, weighted automata, and noncommutative determinants are established
mathematics or methods and must be attributed as such.
