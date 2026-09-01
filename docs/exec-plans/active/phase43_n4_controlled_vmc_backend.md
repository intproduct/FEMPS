# Active execution plan: Phase 43 controlled N>2 explicit-correlation backend

## Objective

Replace the bounded `N=2` `Q^2` truth oracle by a controlled stochastic backend
for the same first-quantized continuous state

```text
Psi(x_1,...,x_N) = exp(sum_(i<j) u_theta(x_i,x_j)) det[phi_a(x_i)],
```

without changing the FEMPS/exterior carrier definition or converting the state
to occupation-number MPS. The first target is estimator validation; no new
Paper B source or physics production point is authorized by this plan.

## Required implementation closure before production

1. Generalize the symmetric Gaussian correlator and functional-basis Slater
   carrier from `N=2` to arbitrary small `N`, initially `N=4` and `chi=1`.
2. Implement coordinate-space log amplitude, drift, local kinetic energy,
   one-body trap and two-body soft-Coulomb energy without enumerating virtual
   paths or materializing a `D^N` coefficient tensor.
3. Add deterministic multi-chain sampling with explicit burn-in, thinning,
   acceptance rate, integrated autocorrelation time, effective sample size,
   blocking uncertainty, chain-to-chain diagnostics, checkpoint and resume.
4. Implement the covariance/log-derivative energy-gradient estimator and
   compare it with deterministic `N=2` quadrature gradients at frozen states.
5. Report sampled particle-swap antisymmetry residual and reject any
   nonsymmetric correlator at the API boundary.
6. Validate the `N=4` noninteracting Slater limit against its exact energy and
   zero-variance expectation before adding interaction.
7. Audit time and memory as functions of `(N,D,P,chains,samples)`; observed
   sampling cost must be separated from mixing/autocorrelation cost.

## Validation gates

- `N=2` fixed-state VMC energy agrees with deterministic quadrature within the
  larger of `5` reported standard errors and a preregistered absolute floor;
- orbital and correlator energy gradients agree with deterministic quadrature
  within combined statistical and finite-difference uncertainty;
- `N=4` noninteracting energy agrees with the exact Slater value within `5`
  standard errors and its local-energy variance is statistically consistent
  with zero at the chosen numerical precision;
- all sampled antisymmetry residuals are at most `1e-12`;
- two clean runs with the same seed/resume boundary reproduce serialized chain
  state and observables exactly, while distinct frozen seeds agree within
  uncertainty;
- no production state may be selected using a reference energy.

Concrete chain counts, seeds, sample budgets, error floors, optimizer budgets,
and the interacting `N=4` comparison must be frozen in a new ADR after the
fixed-state estimator tests, but before the first interacting production run.

## Scientific boundary

Phase 40 establishes only a low-`D`, `N=2` basis-efficiency differentiator.
Phase 43 must determine whether that structure survives controlled stochastic
evaluation at `N=4`. A pass still does not make the Jastrow/determinant form
new. Any later method claim requires an external reproduction and a matched
Li--Waintal or same-basis DMRG comparison that identifies a genuine accuracy,
stability, memory, or complexity tradeoff beyond fixed finite NOCI.
