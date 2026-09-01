# Phase 40 explicit-correlation differentiator report

## Outcome

The preregistered `N=2` gate passes its internal production and reconstruction
criteria. A symmetric Gaussian correlator multiplying one `chi=1` exterior
Slater carrier reaches an approximately `4e-6` absolute reference-error
plateau already at low carrier basis order. Under the frozen primary comparison
against optimized fixed-`K=4` NOCI, `D=2,4,6,8` pass; `D=10,12` do not.

This is evidence for a low-`D` explicit-correlation basis-efficiency tradeoff,
not a monotone all-`D` advantage. It is also not a scalable contraction result:
the correlated `N=2` truth solver explicitly materializes a `Q^2` coordinate
grid. No second manuscript is opened. A clean independent optimization
reproduction remains required by ADR 0030.

## Frozen model and comparison

- two spinless one-dimensional fermions in a unit harmonic trap;
- soft-Coulomb coupling and softening both equal to one;
- relative-coordinate grid reference energy `2.553828651107161`;
- carrier basis `D=2,4,6,8,10,12`;
- nested symmetric pair features `P=0,1,3,5`;
- fixed NOCI controls `K=1,2,4`;
- three fixed correlated seeds and three fixed NOCI seeds per point;
- lowest variational energy across the frozen seeds selected without using the
  reference error as the selector.

The explicitly correlated wavefunction is generally outside
`Lambda^2 V_D`. Same-basis CI is a comparator, not an equal-space variational
bound. The result therefore measures how an explicit real-space correlation
factor changes carrier-basis requirements; it is not evidence that the same
finite-basis variational space was solved more accurately.

## Primary preregistered result

| `D` | best `P=5` error | best `K=4` NOCI error | error ratio | `P=5` raw real parameters | `K=4` nonlinear real parameters | gate point |
|---:|---:|---:|---:|---:|---:|:---:|
| 2 | `4.4796e-6` | `1.1062e-2` | `4.0495e-4` | 9 | 32 | pass |
| 4 | `3.8961e-6` | `4.6008e-4` | `8.4682e-3` | 13 | 64 | pass |
| 6 | `4.1435e-6` | `5.7257e-5` | `7.2367e-2` | 17 | 96 | pass |
| 8 | `4.0975e-6` | `1.3798e-5` | `2.9696e-1` | 21 | 128 | pass |
| 10 | `5.6188e-6` | `7.3739e-6` | `7.6198e-1` | 25 | 160 | fail |
| 12 | `8.3865e-6` | `8.6011e-6` | `9.7505e-1` | 29 | 192 | fail |

The passing consecutive pairs are `(2,4)`, `(4,6)`, and `(6,8)`. Every seed
passes the primary inequality through `D=8`; none passes at `D=10` or `D=12`.
The frozen gate therefore passes without hiding the high-`D` failure.

For the selected `P=5` states, the energy variances at increasing `D` are
`2.198e-5`, `8.096e-6`, `1.011e-5`, `9.744e-6`, `2.185e-5`, and
`4.541e-5`. The corresponding raw norms and quadrature uncertainties are
retained point by point in the machine-readable artifact; raw norms are not
forced to one because the reported energy is the normalized Rayleigh quotient.

The nested feature axis is also informative. At `D=2`, the best errors are
`1.1062e-2`, `5.5789e-3`, `3.2869e-4`, and `4.4796e-6` for `P=0,1,3,5`.
At larger `D`, `P=3` remains near `4.7e-6`, while `P=5` becomes less stable at
`D=10,12`. This outcome-informed observation is not used to change the frozen
primary `P=5` rule or to add a rescue run.

## Validation and resources

- reconstructed correlated points: 72;
- reconstructed NOCI controls: 54;
- maximum serialized/reconstructed correlated observable difference: zero;
- maximum serialized/reconstructed NOCI observable difference:
  `4.441e-15`;
- maximum AD/central-difference discrepancy: `7.585e-11`;
- maximum `Q=128`--`Q=160` energy change: `5.030e-9`;
- maximum relative norm change: `5.504e-11`;
- antisymmetry and correlator-symmetry residuals: zero for every correlated
  point;
- summed correlated optimizer wall time: `120.1 s`;
- summed NOCI optimizer wall time: `499.7 s`;
- maximum sampled process RSS: `0.780 GB` correlated and `1.057 GB` NOCI.
- full repository regression: `292 passed`, with one known latticeTN
  report-path scalar-conversion warning.

The timings describe these particular deterministic CPU implementations and
small matrices. They do not establish asymptotic superiority. The host exposed
CUDA devices, but CPU float64 was frozen for deterministic small-system
comparison.

## Scientific consequence

The result supplies the first internally reconstructed numerical evidence in
this repository for the requested non-NOCI differentiator: one exterior
carrier plus explicit continuous correlation reaches low error without growing
a finite determinant list. The Jastrow form, determinant carrier, and VMC
ideas remain prior art; the result does not itself establish a new ansatz.

Before any Paper-B decision, the same frozen gate must be reproduced from a
clean optimization source without reusing checkpoints. A later many-particle
step would additionally require a controlled stochastic or structured
contraction backend, uncertainty/autocorrelation reporting, and comparison
with Li--Waintal and same-basis DMRG. Until those steps exist, the practical
claim is limited to this bounded `N=2` carrier-basis differentiator.
