# ADR 0032: Preregister Phase 43 fixed-state VMC validation

## Status

Accepted before the first Phase 43 fixed-state validation artifact is run.
This ADR does not freeze or authorize an interacting `N=4` production point.

## Purpose

Validate the coordinate-space sampler, local-energy estimator, uncertainty and
gradient machinery against existing deterministic truth before optimizing or
claiming any new many-particle physics result.

## Frozen validation states

1. `N=2,D=4,P=3`: use the serialized Phase 40 state at correlated seed
   `40001`. This state is selected after Phase 40 as a numerically stable
   implementation-validation fixture; it is not an independently predicted
   physics point.
2. `N=4,D=4,P=0`: canonical lowest-four-orbital Slater in the unit harmonic
   trap with interaction coupling zero. Its exact energy is `8` and its exact
   local-energy variance is zero.

## Frozen sampling budgets

For the `N=2` state, run two clean seeds `43011` and `43012`, each with:

- 32 CPU float64 chains;
- 500 burn-in sweeps;
- 3,000 retained samples per chain;
- 3 full sweeps between retained samples;
- single-particle Gaussian proposal scale `0.8`;
- maximum reported autocorrelation lag `100`;
- checkpoint interval 500 retained samples.

For the `N=4` noninteracting state, use seed `43021`, 16 chains, 100 burn-in
sweeps, 200 retained samples per chain, two thinning sweeps, proposal scale
`0.7`, lag cap 50 and checkpoint interval 40. Force one run to stop after 80
samples per chain and resume it; compare with a clean run.

No replacement seed, altered proposal scale, longer rescue run, or changed
thinning is admitted after results are observed.

## Frozen gates

- each `N=2` VMC energy differs from the `Q=160` deterministic value by at
  most `max(5*reported_standard_error, 2e-4)`;
- the two `N=2` energies agree within
  `5*sqrt(se_1^2+se_2^2)+1e-4`;
- each `N=2` acceptance rate lies in `[0.15,0.85]`, `Rhat<=1.10`, and total
  effective sample size is at least `1000`;
- every checked orbital/correlator gradient component satisfies
  `abs(g_VMC-g_Q160) <= 5*chain_standard_error + 5e-3`;
- `N=4` energy error is at most `1e-12`, variance at most `1e-20`, and the
  resumed and clean samples/observables agree exactly;
- all sampled antisymmetry and correlator-symmetry residuals are at most
  `1e-12`;
- every artifact reports sample count, acceptance, autocorrelation times,
  effective sample size, blocking and chain uncertainty, `Rhat`, time, peak
  RSS, checkpoint lineage, and the absence of `D^N`/virtual-path objects.

## Consequence

A pass authorizes only a new ADR that preregisters an interacting `N=4`
benchmark. It does not authorize Paper B, external-replication language, or a
scalability claim.
