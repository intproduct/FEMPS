# Phase 44 interacting N=4 explicit-correlation D gate

## Decision

The preregistered Phase 44 gate **fails overall**. All six optimizer runs,
forced-resume reproduction, held-out confirmations, confirmation seed
agreements, `D` monotonicity, and the two-consecutive-`D` physical advantage
subgate pass. The frozen selection-quality gate fails and cannot be rescued:

- both `D=4` selection evaluations exceed the `2.5e-4` standard-error limit;
- all four `D=6,8` selection evaluations fall slightly below the 50,000 ESS
  requirement.

No extra sample, altered threshold, replacement lineage, or additional `D`
point is admitted. The result is internal **numerical evidence**, not an
authorization for Paper B, an external replication, a new Jastrow ansatz, or
a scalability/superiority claim.

## Frozen state and reference firewall

The state is a first-quantized continuous `chi=1` exterior Slater carrier
multiplied by a symmetric five-feature Gaussian correlator. The physical model
is four spin-polarized one-dimensional fermions in a unit harmonic trap with
unit soft-Coulomb coupling and softening.

The only preoptimization source is the disclosed Phase 37 `D=6,K=1,seed=3701`
single Slater, isolated in a reference-free fixture. It is truncated, retained,
or zero-padded for `D=4,6,8`; all correlator amplitudes start at zero. The six
optimizations and six selection evaluations complete before the selected
lineages `(2,1,2)` are written and hashed. Only then does the runner import the
separate module that reads the pre-existing `D=14`, CI, and NOCI comparator
artifacts.

No new ordinary NOCI calculation was run.

## Blind selection outcome

| `D` | lineage | energy | reported SE | ESS | R-hat | selection gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 1 | 11.0242889168 | 5.628e-4 | 53,919.8 | 1.000140 | fail: SE |
| 4 | 2 | 11.0240102248 | 5.011e-4 | 53,003.6 | 1.000032 | fail: SE; selected |
| 6 | 1 | 11.0230311439 | 1.232e-4 | 49,732.7 | 1.000005 | fail: ESS; selected |
| 6 | 2 | 11.0231219913 | 8.480e-5 | 47,445.2 | 0.999984 | fail: ESS |
| 8 | 1 | 11.0231330448 | 7.679e-5 | 47,523.5 | 1.000098 | fail: ESS |
| 8 | 2 | 11.0230641011 | 9.243e-5 | 47,622.0 | 1.000134 | fail: ESS; selected |

All acceptance rates are about `0.61`; variances and symmetry diagnostics pass.
The failure is not hidden by the larger confirmation samples.

## Held-out confirmation and physical subgate

| `D` | combined energy | combined SE | absolute error vs D14 | conservative error | K4 NOCI error | conservative ratio | point gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 11.0243089336 | 1.574e-4 | 1.226e-3 | 2.013e-3 | 6.286e-2 | 0.0320 | pass |
| 6 | 11.0231801947 | 3.764e-5 | 9.734e-5 | 2.856e-4 | 7.549e-4 | 0.3783 | pass |
| 8 | 11.0231265435 | 2.623e-5 | 4.369e-5 | 1.749e-4 | 2.015e-4 | 0.8676 | fail |

Here `conservative error = abs(E-E_D14)+5 SE`; the frozen point rule requires
it to be at most half the existing fixed-`K=4` NOCI error. The consecutive
passing pair is `(4,6)`. The combined energies decrease with `D` under the
registered uncertainty-aware monotonicity test. The `D=14` value
`11.023082853674637` remains a finite-basis numerical reference, not a
continuum theorem or bound.

Each confirmation seed separately passes. ESS ranges from 267,436 to 289,054,
R-hat is within `3.1e-5` of one, and reported SE ranges from `3.69e-5` to
`2.32e-4`. Confirmation seed differences are below their preregistered
allowances.

## Symmetry, optimization, and resources

- maximum optimizer-history antisymmetry residual: `1.041e-15`;
- maximum selection/confirmation residual: `9.220e-16`;
- correlator-symmetry residual: zero at all recorded evaluations;
- no amplitude reaches the frozen `[-1,1]` boundary;
- all six 100-step optimizer histories are finite and pass acceptance gates;
- forced `D=6` step-40 interruption/resume and a clean trajectory agree
  bitwise in parameters, histories, positions, RNG state, moments, and proposal
  counts;
- no production path forms a `D^N` coefficient tensor, a full alternating
  tensor, or a virtual-path list.

Individual optimizer runs take 156--178 s. Selection evaluations take 34--39 s
and confirmations 121--138 s in this single accumulating CPU process. The
largest sampled process RSS is 3.80 GB. These bounded measurements are not a
scaling fit.

## Independent verification

Three committed coordinate archives contain every selection and confirmation
sample. Seven small immutable optimizer checkpoints retain the six production
trajectories and the `D=6` clean control. The standalone verifier checks all
source, ledger, archive, artifact, and checkpoint hashes; reconstructs every
state; recomputes all 12 sample observables and gates; and rederives all
aggregate decisions. The maximum recomputed observable difference is zero in
the current environment.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python scripts/verify_phase44_n4_explicit_correlation_d_gate.py
```

## Consequence

The data are meaningful internal evidence that explicit continuous correlation
can reduce low-`D` carrier-basis error at interacting `N=4`, while the `D=8`
fixed-K4 advantage disappears under the conservative uncertainty rule. But
the experiment did not satisfy its complete error-control contract, so the
project must record a failed Phase 44 gate.

There will be no same-point rescue. Subsequent work may only prepare an
independent clean/external reproduction or a separately preregistered matched
Li--Waintal/same-basis-DMRG comparison. It may not start Paper B or describe
the present result as an admitted practical-method success.
