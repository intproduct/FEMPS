# Phase 43 fixed-state coordinate-VMC validation

## Decision

The ADR-0032 fixed-state validation **passes**. The new backend evaluates a
continuous first-quantized symmetric-correlator/exterior-Slater state without
forming a `D^N` coefficient tensor or enumerating virtual paths. This is an
estimator and checkpointing result, not an interacting `N=4` physics result,
an external replication, a scalable-solver claim, or authorization for Paper B.

## Frozen states and budgets

- `N=2,D=4,P=3`: the Phase 40 `seed=40001` state was disclosed in advance as
  a post-result stable implementation fixture. Its deterministic `Q=160`
  energy and AD gradient are the truth oracle.
- VMC seeds are `43011` and `43012`, with 32 chains, 500 burn-in sweeps,
  3,000 retained samples per chain, thinning 3, proposal scale 0.8, and
  maximum autocorrelation lag 100.
- `N=4,D=4,P=0`: the canonical four-orbital noninteracting Slater state uses
  16 chains, 100 burn-in sweeps, 200 retained samples per chain, thinning 2,
  proposal scale 0.7, and seed `43021`. The run is forced to stop after 80
  samples per chain and then resume.
- All calculations use CPU `float64`. The complete frozen sampler
  configurations are serialized in the result artifact.

## Results

The deterministic `N=2` reference energy is
`2.553833552207632`.

| state | VMC energy | absolute error | reported SE | variance | acceptance | ESS | R-hat | antisymmetry residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `N=2`, seed 43011 | 2.553849311250104 | 1.576e-5 | 1.247e-5 | 1.346e-5 | 0.62165 | 86,550.0 | 1.000015 | 3.83e-16 |
| `N=2`, seed 43012 | 2.553820002964781 | 1.355e-5 | 1.245e-5 | 1.294e-5 | 0.62242 | 83,452.7 | 1.000006 | 3.83e-16 |
| `N=4`, noninteracting | 8.000000000000000 | 0 | 0 | 4.96e-31 | 0.58334 | 3,200.0 | 1.000000 | 7.19e-16 |

The two `N=2` estimates differ by `2.931e-5`, below the frozen
`1.881e-4` allowance. Correlator-symmetry residuals are zero. The maximum
stored VMC-versus-deterministic gradient-component difference is
`1.064e-4`, below its frozen chain-error-plus-`5e-3` allowance. Both energy
points, every gradient component, acceptance, ESS, R-hat, and symmetry gates
pass.

The `N=4` partial run stops at the registered boundary. Resumed and clean runs
have identical coordinate samples and serialized observables, and the
noninteracting local energy is 8 to machine precision with zero numerical
variance at the registered tolerance.

Peak process RSS during the two `N=2` runs was 684 MB and 691 MB; elapsed
times were 16.1 s and 15.5 s. These are two bounded workstation measurements,
not a scaling fit.

## Independent artifact verification

The result JSON hashes the implementation, benchmark, ADR, and compressed raw
coordinate archive. The standalone verifier:

1. checks all hashes and exact archived array shapes;
2. reconstructs the disclosed Phase 40 carrier/correlator state;
3. recomputes the `Q=160` energy and AD gradient;
4. recomputes all `N=2` VMC observables, uncertainty diagnostics, gradients,
   and gate decisions from the archived coordinates;
5. recomputes the `N=4` observables; and
6. independently repeats the forced `N=4` interruption/resume and clean run.

All recomputed observable and gradient differences are exactly zero in the
current environment.

## Limitations and next gate

The `N=2` fixture is close to an optimized stationary state. Its gradient
comparison is therefore an absolute-error implementation check, not evidence
of accurate relative gradients away from stationarity. The result also says
nothing yet about interacting `N=4` mixing, optimization stability, variance,
or `D` convergence.

Before any interacting production run, a new ADR must freeze the physical
model, `D` and correlator axes, reference-use firewall, optimizer and sampler
budgets, seeds, failure rules, checkpoint contract, uncertainty/ESS/R-hat
thresholds, antisymmetry tolerance, and the matched NOCI or same-basis control.
The production question must be the registered non-NOCI differentiator: an
explicit-correlation `D`-convergence advantage or a matched Li--Waintal/
same-basis-DMRG tradeoff. A failed gate is to be retained without a rescue
point.

## Reproduction

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python scripts/verify_phase43_fixed_state_vmc_validation.py
```

Authoritative artifacts:

- `docs/experiments/results/phase43_fixed_state_vmc_validation.json`
- `docs/experiments/results/phase43_fixed_state_vmc_samples.npz`
- `docs/decisions/0032-preregister-phase43-fixed-state-vmc-validation.md`
