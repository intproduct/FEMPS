# Phase 28 diagonal-path FEMPS ladder report

## Evidence level and scope

All results below are **numerical evidence** from the exact, restricted
nonbranching diagonal-path FEMPS route. They do not establish that generic
FEMPS is efficiently contractible or that this method outperforms CI, AGP, or
DMRG. The production observable path uses `K^2` determinant transitions; full
exterior materialization is invoked separately only as a bounded truth audit.

Reproduction command:

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_phase28_diagonal_path_ladder.py --steps 200 --seed 17 --device cpu
```

The raw artifact is
`docs/experiments/results/phase28_diagonal_path_ladder.json`. Ignored checkpoint
files are regenerated under `checkpoints/phase28_diagonal_path_ladder/`.

## Results

The interacting Hamiltonian is the analytic all-to-all harmonic model at
`N=2`, `kappa=0.35`, whose continuum ground energy is
`2.455760721560795`.

| Gate/point | Energy | Error vs same-`D` truth | Error vs continuum | Variance | Norm error | Exterior difference | Antisym. residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 `N=2,D=6,K=1` | 2.000000000000 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2 `D=4,K=4` | 2.456770326271 | 3.860e-12 | 1.010e-3 | 8.386e-12 | 1.110e-16 | 2.220e-15 | 0 |
| E2 `D=6,K=1` | 2.456043094957 | 2.718e-4 | 2.824e-4 | 1.296e-3 | 1.110e-16 | 0 | 0 |
| E2 `D=6,K=2` | 2.455819990378 | 4.874e-5 | 5.927e-5 | 3.393e-4 | 2.220e-16 | 1.332e-15 | 0 |
| E2 `D=6,K=4` | 2.455773778831 | 2.525e-6 | 1.306e-5 | 2.624e-5 | 2.220e-16 | 3.997e-15 | 0 |
| E2 `D=8,K=4` | 2.455770607430 | 9.797e-6 | 9.886e-6 | 8.423e-5 | 2.220e-16 | 0 | 0 |
| E3 `N=4,D=6,K=1` | 8.000000000000 | 0 | 0 | 0 | 0 | 0 | 0 |
| E4 `D=6,K=1` | 12.347908076639 | 8.808e-2 | 2.290e-1 | 3.238e-1 | 0 | 0 | 0 |
| E4 `D=6,K=2` | 12.260541168923 | 7.173e-4 | 1.416e-1 | 7.162e-3 | 0 | 8.882e-15 | 0 |
| E4 `D=6,K=4` | 12.260296106156 | 4.722e-4 | 1.413e-1 | 4.358e-3 | 2.220e-16 | 1.776e-14 | 0 |
| E4 `D=5,K=4` | 12.590372409939 | 2.490e-7 | 4.714e-1 | 7.581e-7 | 0 | 1.776e-15 | 0 |
| E4 `D=7,K=4` | 12.176252401718 | 3.269e-3 | 5.730e-2 | 3.567e-2 | 2.220e-16 | 8.882e-15 | 0 |

The table reports the materialized residual; the structural antisymmetry
residual is also exactly zero for every point. CPU optimization time for the
twelve points is recorded individually in the raw artifact and ranges from
`0.014 s` to `22.57 s`. CPU peak memory is not yet instrumented, so its field is
explicitly null rather than inferred.

A separate Blackwell parity point repeats E4 `N=4,D=6,K=4` on
`NVIDIA RTX PRO 4000 Blackwell`: the CPU/GPU energy difference is
`7.75e-13`, peak allocated GPU memory is `30,736,384` bytes, and elapsed time is
`56.31 s`. The tiny determinant-minor loops are slower on GPU than CPU; this is
a resource/parity check, not an acceleration result. Raw data are in
`phase28_diagonal_path_gpu_parity.json`.

## Interpretation

- E1 passes exactly at `K=1`, confirming that exchange statistics do not force
  a larger correlation path count in this restricted FEMPS representation.
- At fixed `D=6`, increasing `K=1,2,4` lowers both the variational error and
  energy variance systematically. This is the first runnable interacting
  correlation-multiplicity closure for the recovery stage.
- At fixed `K=4`, the energy improves from `D=4` through `D=6` to `D=8`.
  The `D=8` variance is higher than at `D=6`, so higher-basis optimization is
  not yet considered fully conditioned; no monotone-variance claim is made.
- Every production energy agrees with the independently materialized exterior
  Hamiltonian to at most `4.0e-15`, and checkpoint/resume equivalence is covered
  by an automated test.
- E3 passes at `K=1`, while the independently materialized ordinary particle-TT
  ranks are `(4,6,4)`. This is the required exchange-versus-correlation
  representation check, not a runtime advantage claim.
- The E4 pilot has two independent trends. At `D=6`, same-basis error and
  variance decrease from `K=1` through `K=2` to `K=4`. At `K=4`, the continuum
  energy error decreases from `D=5` through `D=6` to `D=7`. The rising
  same-basis optimizer error/variance at `D=7` shows that nonlinear convergence
  is not uniform in `D`, so E4 is not yet a final performance benchmark.

## Decision

E1--E3 pass, and the first controlled E4 pilot is complete. The E3 artifact
records the ordinary TT exchange ranks; the already proved flat spectrum is
covered by the existing theorem regression and will be copied into the final
benchmark comparison table. E4 remains open pending stronger `D=7`
optimization stability, CPU peak-memory instrumentation, and the required
CI/ordinary-TT/AGP comparator table. In parallel, the diagonal transition path
still needs a faster well-conditioned implementation; the singular-safe
determinant-minor route remains the correctness reference.

This pilot decision is superseded by the independently verified closure in
`phase28_e4_closure_report.md`: all three missing items now pass, the guarded
inverse path retains automatic minor fallback, and E4 is accepted at the
registered restricted-algorithm evidence level.
