# Phase 39 report: distinctiveness audit and correlated-carrier prototype

## Decision

Phase 39 selects one candidate and no backup ansatz: a symmetric explicit
correlator multiplying an exterior functional carrier. Li--Waintal and
same-basis DMRG remain mandatory comparators, not candidate FEMPS methods. ADR
0029 freezes the state boundary and the next falsifiable gate.

## Why the alternatives are insufficient

- The completed diagonal-path solver is finite NOCI.
- LC-AGP/Pfaffian sums have direct prior art and do not supply the missing
  differentiator.
- The project's ordered COM/gap solver is close to Li--Waintal and cannot be
  renamed FEMPS.
- Recent tensor backflow work already covers lattice/Fock-space tensorized
  backflow and second-quantized CP-backflow with VMC/DMRG comparison. A broad
  tensor-backflow novelty claim is unavailable.

The surviving question is narrower: can a continuous explicit correlator be
coupled to an exterior carrier with independent functional-basis and
correlation controls, and does that coupling produce a measured convergence or
cost advantage over fixed-`K` NOCI?

## Exploratory bounded prototype

The implementation in `femps.algorithms.correlated_exterior` supports:

- normalized harmonic functional values and analytic first derivatives;
- a two-fermion exterior Slater carrier;
- symmetric Gaussian pair correlators;
- deterministic norm, kinetic, trap, soft-Coulomb energy, and energy variance;
- automatic differentiation and finite-difference comparison;
- explicit antisymmetry and correlator-symmetry residuals; and
- bounded projection into `Lambda^2 V_D` to audit Slater rank.

The materialization uses `Q^2` coordinate values and is explicitly not a
production contraction.

## Exploratory numerical evidence

At `N=2`, coupling one and softening one, a five-feature correlator multiplying
the canonical lowest two harmonic orbitals gives:

| Quantity | Value |
|---|---:|
| Correlated energy, Q160 | `2.553833129442` |
| Same uncorrelated carrier | `2.564890847246` |
| Independent relative-grid reference | `2.553828651107` |
| Absolute reference error | `4.478e-6` |
| Energy variance | `2.136e-5` |
| Q128-to-Q160 relative norm change | `5.503e-11` |
| AD/finite-difference gradient difference | `1.509e-11` |
| Antisymmetry residual | `0` |
| Correlator symmetry residual | `0` |

The projected antisymmetric matrix is full rank at every audited even order:

| Projection order `D` | Matrix rank | Slater rank |
|---:|---:|---:|
| 4 | 4 | 2 |
| 6 | 6 | 3 |
| 8 | 8 | 4 |
| 10 | 10 | 5 |
| 12 | 12 | 6 |

This is evidence that the fixed correlated carrier is not a single Slater and
requires an increasing determinant count in these finite projections. It is
not a general rank theorem and does not yet establish a matched-cost advantage
over optimized NOCI.

The same-basis CI energies at `D=2,4,6,8,10,12` are respectively
`2.564890847246`, `2.554288734101`, `2.553885908134`, `2.553842079144`,
`2.553834336261`, and `2.553832510344`. The correlated carrier uses functions
outside each finite `Lambda^2 V_D`, so this is an explicit-correlation basis
comparison, not an equal-variational-space claim.

## Resource and evidence boundary

The exploratory call completed in seconds, materialized at most `160^2`
coordinate values, and used dense CI only as bounded truth. It enumerated no
FEMPS virtual paths. Runtime and sampled RSS are machine-dependent fields in
the JSON artifact and are not scaling evidence.

The result closes only the definition/materialization/AD prerequisite. The
next active phase must preregister the matched `D/P/K` comparison before
viewing its production result. No second manuscript is opened.
