# ADR 0033: Preregister the interacting N=4 explicit-correlation D gate

## Status

Accepted on 2026-09-02 after ADR-0032 fixed-state validation and before any
interacting `N=4` coordinate-VMC production result.

## Scientific question

Test exactly one next claim: whether a fixed `P=5`, `chi=1` symmetric explicit
correlator plus continuous exterior Slater carrier shows a reproducible carrier-
basis `D`-convergence advantage over already-computed fixed-`K=4` NOCI data.
This is the differentiator required by the single-manuscript decision. The
Jastrow determinant ansatz and VMC optimizer are prior art and are not claimed
as new.

## Frozen physical model and axes

- four spin-polarized fermions on the real line;
- unit harmonic trap and soft-Coulomb coupling/softening `g=a=omega=1`;
- real harmonic functional carrier basis `D in {4,6,8}`;
- one exterior Slater carrier, so `chi=1` throughout;
- Gaussian pair exponents `{0.0625,0.25,1,4,16}`, hence fixed `P=5`;
- no additional ordinary NOCI point, particle number, `D`, `P`, or ansatz is
  admitted after outcomes are observed.

The multiplied state is generally outside `Lambda^4 V_D`. Same-basis CI and
NOCI are therefore comparators, not equal-variational-space bounds.

## Frozen initialization and disclosure

Use the pre-existing Phase 37 `N=4,D=6,K=1,seed=3701` optimized single-Slater
orbital matrix from
`docs/experiments/results/phase37_slater_source_solver.json`:

- truncate its first four rows and QR-retract for `D=4`;
- use it directly, after taking its real part and QR-retracting, for `D=6`;
- append two zero rows and QR-retract for `D=8`.

All five correlator amplitudes start at zero. This is a disclosed historical
single-Slater preoptimization source, not a clean canonical initialization and
not an FCI/NOCI multideterminant initialization. No state is initialized from
the `K=2,3,4` Phase 37 stages or from the `D=14` reference.

## Frozen stochastic optimizer

At each `D`, run two independent lineages with seeds:

| `D` | lineage 1 | lineage 2 |
|---:|---:|---:|
| 4 | 44041 | 44042 |
| 6 | 44061 | 44062 |
| 8 | 44081 | 44082 |

Every lineage uses CPU `float64`, 32 chains, 100 Adam/QR-retraction updates,
1,000 initial burn-in sweeps, 20 rethermalization sweeps per later update, 128
retained samples per chain per update, thinning 2, proposal scale `0.65`,
autocorrelation lag cap 100, learning rate `0.01 -> 0.001`, gradient-norm clip
2, amplitude box `[-1,1]`, and checkpoint interval 10. Adam uses
`beta1=0.9`, `beta2=0.999`, and epsilon `1e-8`.

Force `D=6`, lineage 1 to stop after update 40 and resume. Also run that exact
lineage clean and require bitwise-identical parameters, history, proposal
counts, and final state. No altered seed, learning rate, bound, or rescue
budget is admitted.

## Frozen state selection and confirmation

Evaluate the two completed lineages at each `D` with independent selection
seeds `45041/45042`, `45061/45062`, and `45081/45082`, respectively. Each
selection evaluation uses 32 chains, 1,000 burn-in sweeps, 2,000 retained
samples per chain, thinning 3, proposal scale `0.65`, and lag cap 200. Select
the lower reported VMC energy without loading any reference or NOCI error.

The selected state is immutable. Confirm it with two held-out seeds:

| `D` | confirmation seed 1 | confirmation seed 2 |
|---:|---:|---:|
| 4 | 44241 | 44242 |
| 6 | 44261 | 44262 |
| 8 | 44281 | 44282 |

Each confirmation uses 64 chains, 2,000 burn-in sweeps, 5,000 retained
samples per chain, thinning 4, proposal scale `0.65`, and lag cap 500. Report
each seed separately and combine only the two estimates of the same immutable
state by inverse-variance weighting. The confirmation result may not trigger a
lineage switch.

## Reference firewall and frozen comparators

The production runner may load initialization orbitals but must not load the
following reference/comparator values until all six optimization checkpoints,
six selection evaluations, and all three lineage choices have been serialized
and hashed.

The pre-existing `D=14,Q=128` exterior value
`11.023082853674637` is a numerical reference, not a continuum theorem or
bound. Frozen same-basis CI energies are:

| `D` | CI energy |
|---:|---:|
| 4 | 11.085944151108343 |
| 6 | 11.023837713203346 |
| 8 | 11.023278984749750 |

Frozen fixed-`K=4` NOCI comparator energies and absolute `D=14` errors are:

| `D` | NOCI energy | absolute reference error | source |
|---:|---:|---:|---|
| 4 | 11.085944151108343 | 6.286129743370594e-2 | one-dimensional `Lambda^4 V_4` truth |
| 6 | 11.023837713691630 | 7.548600169933195e-4 | Phase 37 clean K4 |
| 8 | 11.023284391447700 | 2.015377730621992e-4 | Phase 10 K4 hierarchy |

The verifier must re-read these source artifacts and reject changed values or
hashes. No new NOCI optimization is run.

## Frozen validation and success gates

Every optimization and evaluation must report acceptance, energy variance,
ESS, blocking/chain standard errors, R-hat, elapsed time, peak RSS,
antisymmetry residual, correlator-symmetry residual, checkpoint identity,
parameter counts, and zero `D^N`/full-alternating-tensor/virtual-path counts.

The gate passes only if all conditions hold:

1. all histories and final tensors are finite; every recorded acceptance rate
   is in `[0.15,0.85]`; no amplitude equals the frozen box boundary;
2. the forced `D=6` resume and clean trajectory are bitwise identical;
3. every selection and confirmation run has R-hat at most `1.05`, ESS at least
   50,000, reported standard error at most `2.5e-4`, variance at most `0.02`,
   and both symmetry residuals at most `1e-12`;
4. the two confirmation seeds at each `D` agree within
   `5*sqrt(se1^2+se2^2)+2e-4`;
5. combined selected-state energies are nonincreasing with `D` within
   `5` combined standard errors plus `2e-4`;
6. a point passes the differentiator when
   `abs(E_corr-E_D14)+5*se_corr <= 0.5*abs(E_NOCI(K=4)-E_D14)`;
7. point 6 passes at two consecutive values in the frozen `D` axis; and
8. the standalone verifier recomputes observables from hashed raw coordinates
   and reproduces every selection and aggregate decision.

The error-ratio rule is deliberately conservative about VMC uncertainty. A
failure is retained as negative numerical evidence. Thresholds, states, seeds,
or axes may not be changed to rescue it.

## Publication consequence

A pass supplies internal interacting evidence for the requested explicit-
correlation `D` differentiator. It does not by itself authorize Paper B,
novelty, scalability, or superiority language. Before a separate methods paper
is opened, the passing result requires clean repository reproduction and then
external replication or a matched Li--Waintal/same-basis-DMRG comparison with
a measured accuracy, stability, time, or memory tradeoff. The combined paper
remains the only manuscript while those conditions are unmet.
