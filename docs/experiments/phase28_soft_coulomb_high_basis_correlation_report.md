# Phase 28 high-basis soft-Coulomb correlation audit

## Question and boundary

The accepted `N=4,D=12,K=4` soft-Coulomb point has a same-basis dense-CI
error of `8.314e-5`. This audit asks whether that residual is systematically
reducible by the FEMPS correlation control `K`, rather than being hidden inside
the functional-basis error.

This is **numerical evidence** from one seeded `K=4 -> 5` growth point, not a
claim about asymptotic `K` scaling. The first four Slater orbital sets are
preserved exactly and one seed-2812 blind random Slater is added. Variable
projection therefore starts from a span containing the accepted `K=4` state;
no CI eigenvector selects or initializes the new term.

Reproduce and verify with

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_phase28_soft_coulomb_high_basis_correlation.py
.\.venv\Scripts\python.exe scripts\verify_phase28_soft_coulomb_high_basis_correlation.py
```

Raw data are in
`results/phase28_soft_coulomb_high_basis_correlation.json`.

## Preregistered decisions

The audit gate retained all earlier state and operator thresholds. In addition:

- the initial `K=5` span could not worsen the source `K=4` energy by more than
  `1e-9`;
- optimization could not worsen that nested initial energy by more than
  `1e-9`;
- `material_improvement` required at least a 10% reduction of same-basis
  dense-CI error and nonincreasing variance.

Audit completion and material improvement were separate decisions, so a stable
but unhelpful fifth determinant would have been recorded as such.

## Result

| `K` | Energy | Same-basis dense-CI error | Variance | Center particle-TT rank |
|---:|---:|---:|---:|---:|
| 4 | 11.023177795943152 | `8.313953e-5` | `9.014082e-4` | 24 |
| 5 | 11.023140923997320 | `4.626759e-5` | `5.290684e-4` | 30 |

The error reduction is 44.35%, above the preregistered 10% threshold, and the
variance falls by 41.31%. The nested initial energy is
`11.023177768091395`, already nonworsening because the blind fifth determinant
enlarges the variational span; subsequent optimization lowers it further.

The final overlap condition number is 4.152 with all five determinant
directions retained. Norm error and both structural and materialized
antisymmetry residuals are zero. Production enumerates zero virtual paths.
The direct `D=12` exterior CI energy is `11.023094656411180` and its center
particle-TT rank is 66.

## Cost and interpretation

The `K=5` call took 141.91 s and sampled peak process RSS was 827,375,616
bytes. Compared with the `K=4` D12 call at 92.77 s, the measured time ratio is
1.53, close to the expected transition-count ratio `25/16=1.5625`. Peak RSS
changes little in absolute terms, while the correlation-dependent center
particle-TT rank grows from 24 to 30.

This is the intended FEMPS separation in a measured bounded example: exterior
construction keeps exchange antisymmetry exact, while increasing `K` changes
the correlation approximation and systematically reduces its error. It is a
clear structural advantage/tradeoff, not a demonstrated runtime advantage:
dense CI remains much faster at this small validation dimension, and the
production cost grows as `K^2 L`.

## Decision

Both `audit_pass` and `material_improvement` pass under independent
verification. The fixed-`K` and functional-basis errors are now explicitly
separated at high basis order. Together with the fresh-process D8-to-D12
lineage, this completes the bounded Phase 28 consolidation gate.

The next project decision may consider a resource-capped `N=6` interacting
pilot, but only after freezing the present solver API and reproducibility
contract. Such a pilot must use an independently controlled reference or error
bound and may not revive generic scalability language.
