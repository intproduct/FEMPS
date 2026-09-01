# ADR 0029: Select an explicit-correlator exterior carrier for the next gate

## Status

Accepted on 2026-09-02 after the bounded Phase 39 route audit.

## Context

The nonbranching diagonal-path implementation is exactly a finite NOCI
expansion and cannot support an independent method claim. The ordered-distance
route is an implementation-level continuation of Li--Waintal and must retain
that name. Same-orbital-basis DMRG is a required comparator, not a FEMPS state.

Tensor/backflow and Jastrow ansatzes are established prior art. Nevertheless, a
symmetric explicit correlator multiplying a continuous exterior carrier gives
one falsifiable way to test the project's intended separation of exchange and
correlation without returning to a fixed finite determinant list.

## Decision

Select exactly one main candidate:

```text
Psi_(theta,A) = J_theta Psi_A^exterior,
J_theta = exp(sum_(i<j) u_theta(x_i,x_j)),
u_theta(x,y) = u_theta(y,x).
```

Start with a `chi=1` exterior Slater carrier and a finite continuous pair-
feature basis. The permutation-symmetric multiplier preserves exact
antisymmetry. `D` controls the carrier functional basis, `P` controls explicit
correlation features, and `chi` remains the exterior-carrier multiplicity. The
uncorrelated limit is `P=0` or zero correlator amplitudes.

This is explicitly a carrier--correlation extension. For finite `D`, the full
multiplied wavefunction is generally not contained in `Lambda^N V_D`; same-
basis CI is therefore a comparator, not the same variational space. Every
result must disclose this rather than presenting the comparison as equal-basis
variational superiority.

## Contraction route

- `N=2`: deterministic product quadrature is admitted only as a bounded
  materialization, gradient, and projection-rank oracle.
- Larger `N`: the intended route is VMC or a hybrid deterministic--stochastic
  estimator with reported variance, autocorrelation, effective sample size,
  failure probability, and Rayleigh-quotient uncertainty.
- Generic exact matrix-wedge contraction remains obstructed and is not assumed.
- Every approximation reports an antisymmetry residual. A nonsymmetric
  correlator is rejected by construction/API tests.

No backup ansatz is activated. Li--Waintal and same-basis DMRG are mandatory
comparators. Existing NOCI/CI code remains a control.

## Prior-art boundary

The Jastrow factor, backflow idea, determinant carrier, VMC engine, and tensor
decomposition are not claimed new. Zhou--Zhou--Liang (2024) and
Bortone--Rath--Booth (2025) already tensorize backflow, the latter in a
second-quantized CP/VMC setting with DMRG comparisons. The only admissible
future contribution is a demonstrated advantage or tradeoff from the
first-quantized continuous functional/exterior integration and its independently
controlled `D`, `P`, and `chi` axes.

## Next falsifiable gate

Before any method-paper writing, preregister an `N=2` soft-Coulomb comparison
with fixed seeds/budgets and no outcome-dependent retries:

1. independently vary carrier basis `D` and correlator feature count `P`;
2. compare with optimized fixed-`K` NOCI, same-basis CI, the relative-coordinate
   reference, and the existing ordered-coordinate control where matched;
3. report energy error, variance, quadrature/statistical uncertainty, raw and
   normalized norm checks, antisymmetry residual, AD gradient error, time, and
   peak RSS;
4. test whether projected Slater rank grows with projection `D` while the
   correlated representation remains fixed;
5. reject the route if it does not show a reproducible `D`-convergence or
   matched-cost advantage beyond fixed-`K` NOCI.

Passing the exploratory materialization prototype does not pass this gate and
does not reopen a second manuscript.
