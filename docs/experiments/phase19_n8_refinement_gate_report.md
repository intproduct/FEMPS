# Phase 19 report: resource-safe local optimization and N=8 Gate G

## Outcome

Gate G is **PASS (resource-closed controlled N=8 point)**. The two explicit
Gate F qualifications are resolved without deleting their historical evidence:

1. the chi-32 two-site local solver now stays below the predeclared 2 GiB CUDA
   budget after replacing one pathological five-operand contraction path;
2. MPO bond 128 passes a left-gauge physical-tangent audit against bond 192,
   while the earlier raw tensor-parameter threshold remains recorded as a
   failed auxiliary Gate F diagnostic;
3. matched bond-128/160/192 training selects 128 as the smallest passing
   production bond under the declared energy, spread, and memory budgets;
4. a blind `N=8,D=12` multiscale run improves the error against the same
   exterior `D=14` numerical reference by `17.6%` relative to Gate F's D=10
   point; and
5. Fourier, local quadrature, basis conditioning, independent chi-32 local
   optimization, and N=2/4/6/8 resource-trend controls all close.

This decision admits one controlled N=8 point. It does not establish a
continuum error bound, favorable asymptotic scaling, or N=10 admission. The
ordered-distance functional-TN branch remains distinct from FEMPS.

## Resource-safe chi-32 effective Hamiltonian

The previous two-site matrix-free action used one five-operand einsum. The
selected contraction path requested a 78.12 GiB temporary at
`N=8,D=10,chi=32,W=128`. Phase 19 evaluates the same action as four explicit
two-operand contractions:

```text
L x Theta -> left-dressed Theta
           -> apply first MPO tensor
           -> apply second MPO tensor
           -> contract R.
```

Every staged temporary is bounded by explicit environment, MPS, MPO, and local
basis dimensions. At the Gate F point the largest analytic bound is
`chi^2 W D^2 = 13,107,200` float64 entries, or 100 MiB. A small direct dense
action and all existing latticeTN DMRG tests agree with the staged path.

A formal seed-1862 Gate F state reproduction followed by two chi-32 Lanczos
sweeps gives:

| Quantity | Result | Budget |
|---|---:|---:|
| Initial global-AD energy | `44.454373263575` | diagnostic |
| Final local energy | `44.454321906013` | nonincrease |
| Two-sweep energy difference | `3.684e-11` | `<1e-6` |
| Peak CUDA memory | 736,884,736 bytes | `<2,147,483,648` |
| Local elapsed time | 27.84 s | recorded |

The same staged path later runs at `N=8,D=12,chi=32`, peaking at
1,069,710,336 bytes. The local solver qualification is therefore closed at
both the Gate F and refined Gate G points.

## Gauge-aware MPO derivative audit

Raw MPS tensors contain gauge directions, so a componentwise raw-parameter
gradient comparison is coordinate dependent. Phase 19 left-canonicalizes the
fixed state and constructs deterministic one-site directions satisfying
`A^T B=0` on every nonfinal nonzero tangent block. The final-site direction has
its state component removed. Each resulting many-body tangent is normalized
with the native MPS norm; maximum state overlap and gauge residuals are checked
explicitly. Autograd directional derivatives have an independent centered
finite-difference unit test.

Fourteen fixed physical directions are evaluated under MPO bonds 128, 160,
and 192. Against bond 192:

| Bond | Energy difference | Max tangent difference | Relative tangent L2 | Cosine | Pass |
|---:|---:|---:|---:|---:|:---:|
| 128 | `3.597e-7` | `3.941e-8` | `2.022e-5` | `0.999999999811` | yes |
| 160 | `3.254e-10` | `1.045e-11` | `3.632e-9` | `1.000000000000` | yes |

The predeclared bond-128 budgets are `1e-6` for energy and maximum tangent
difference, `5e-5` for relative tangent L2, and `0.99999999` minimum cosine.
Bond 128 passes all four. This complements rather than rewrites Gate F: its
raw parameter-gradient relative difference remains `2.90e-5` versus the extra
`2e-6` threshold and is still labeled a historical auxiliary miss.

## Matched MPO-bond training

Bond choices were also trained independently with the same seed 1862,
four-stage 1,600-step schedule, MPS bond 32, and all other physics controls
fixed. Every final state is re-evaluated under bond 192.

| Training bond | Own-MPO energy | Bond-192 evaluation | Own/ref difference | Time | Peak CUDA |
|---:|---:|---:|---:|---:|---:|
| 128 | `44.4543732636` | `44.4543729039` | `3.597e-7` | 127.11 s | 523,293,696 B |
| 160 | `44.4543813760` | `44.4543813756` | `3.254e-10` | 169.24 s | 648,261,632 B |
| 192 | `44.4543950093` | `44.4543950093` | zero | 224.97 s | 765,400,064 B |

The bond-192 evaluation spread across independently trained states is
`2.211e-5`, below the declared `2e-4` optimization-spread budget. All
own-versus-reference differences are below `1e-6` and every run remains below
2 GiB. Bond 128 is retained as the smallest passing production value; bond 192
is an audit reference, not automatically a better optimizer outcome.

## Stronger exterior numerical reference

The independent Slater--Condon exterior Hamiltonian is extended from D=12 to
D=14. Its dimension is `binom(14,8)=3003`.

| Basis/order | Ground energy | Hermiticity residual | Time |
|---|---:|---:|---:|
| D=12, Q=160 | `44.446009528435` | `1.586e-18` | 26.08 s |
| D=14, Q=128 | `44.445670415298` | `1.980e-18` | 263.16 s |
| D=14, Q=160 | `44.445670415299` | `2.093e-18` | 262.37 s |

The D14 Q128/Q160 difference is `9.24e-13`; the D12-to-D14 shift is
`3.391e-4`. D14 Q160 is therefore the strongest available finite-basis
numerical reference. It is not a D-to-infinity or continuum bound, and no D16
dense exterior calculation is claimed.

## Blind N=8,D=12 refinement

Ten `(ell,rho)` candidates were fixed before reference evaluation:
`ell in {0.40,0.45,0.50,0.55,0.60}` and `rho in {2.5,3.0}`. Each receives the
same 250-step, chi-16 scan with seed 1940. The minimum scan energy selects
`(ell,rho)=(0.55,3.0)`, an interior scale point rather than a search-window
edge. Production then uses a new seed 1941, chi 32, D=12, `M=96`, `Q=224`, MPO
bond 128, and the declared four-stage 1,600-step schedule.

| Control | Result | Budget |
|---|---:|---:|
| Production energy | `44.452844223285` | diagnostic |
| Error vs exterior D14 Q160 | `7.174e-3` | `<1.2e-2` |
| Gate F D10 error vs same D14 reference | `8.703e-3` | comparator |
| Error reduction from D10 | `17.57%` | improvement required |
| Independent chi-32 local-optimizer difference | `5.614e-5` | `<1e-3` |
| Fixed-state M96/M112 difference | `2.824e-6` | `<5e-6` |
| Fixed-state Q192/Q224 difference | `3.162e-12` | `<1e-8` |
| Production peak CUDA memory | 768,552,960 bytes | `<2 GiB` |
| Local-optimizer peak CUDA memory | 1,069,710,336 bytes | `<2 GiB` |

An independent 1,000-point overlap audit gives a maximum orthonormality
residual `4.735e-11`; the collision-boundary residual is zero. The primitive
overlap condition number is `7.535e4` and remains an exposed numerical
control.

## Accuracy-to-resource reassessment

Representative best controlled points are compared below. N=2/4 use bounded
same-basis truth solvers, while N=6/8 use production global AD, so time and
memory columns are descriptive rather than a matched-solver scaling fit.

| N,D | Absolute reference error | Stored MPO elements | Error/element diagnostic | Time | Peak CUDA |
|---:|---:|---:|---:|---:|---:|
| 2,10 | `2.018e-7` | 1,000 | `2.018e-10` | not separately recorded | CPU |
| 4,8 | `8.977e-4` | 532,480 | `1.686e-9` | 0.569 s | CPU |
| 6,10 | `2.470e-3` | 6,218,000 | `3.973e-10` | 71.65 s | 424,656,896 B |
| 8,12 | `7.174e-3` | 14,192,640 | `5.055e-10` | 159.35 s | 768,552,960 B |

From the N=6 to N=8 production point, stored MPO entries, time, and peak CUDA
memory increase by factors `2.28`, `2.22`, and `1.81`; the absolute errors
against their separate finite-basis references differ by a factor `2.90`.
Because local orders, references, and small-N solvers differ, none of these
ratios is an asymptotic exponent. The trend explicitly defers N=10.

## Decision and limits

Gate G closes the local-solver resource and physically meaningful MPO-gradient
qualifications at N=8 and admits the blind D12 refinement. The production
defaults remain MPO bond 128 and chi 32 for this controlled point.

The following are not inferred:

1. a continuum or D-to-infinity error bound;
2. favorable end-to-end scaling in particle number;
3. N=10 feasibility or accuracy;
4. method priority over Hong et al. or Li--Waintal; or
5. identification of the ordered-distance route with FEMPS.

## Reproduction

```powershell
.\.venv\Scripts\python scripts\benchmark_phase19_dmrg_memory.py --device auto
.\.venv\Scripts\python scripts\benchmark_phase19_mpo_tangent.py --device auto
.\.venv\Scripts\python scripts\benchmark_phase19_mpo_bond_training.py --device auto
.\.venv\Scripts\python scripts\benchmark_phase19_n8_d12_basis.py --device auto
.\.venv\Scripts\python scripts\build_phase19_resource_trend.py
.\.venv\Scripts\python scripts\sweep_soft_coulomb_many_body_truth.py --particles 8 --basis-orders 14 --quadrature-orders 128 --production-quadrature 128 --output tmp\soft_coulomb_n8_d14_q128.json
.\.venv\Scripts\python scripts\sweep_soft_coulomb_many_body_truth.py --particles 8 --basis-orders 14 --quadrature-orders 160 --production-quadrature 160 --output tmp\soft_coulomb_n8_d14_q160.json
.\.venv\Scripts\python scripts\build_phase19_n8_reference.py --q128 tmp\soft_coulomb_n8_d14_q128.json --q160 tmp\soft_coulomb_n8_d14_q160.json
```

The two isolated D14 exterior runs take several minutes each. Formal records
are:

- `results/phase19_dmrg_contraction.json`;
- `results/phase19_mpo_tangent_audit.json`;
- `results/phase19_mpo_bond_training.json`;
- `results/phase19_n8_exterior_d14.json`;
- `results/phase19_n8_d12_multiscale.json`; and
- `results/phase19_accuracy_resource_trend.json`.
