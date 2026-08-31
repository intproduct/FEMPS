# Gate A report: fixed-number Pfaffian subclass

## Decision

**CONDITIONAL PASS.** Generic matrix-wedge FEMPS remains without a polynomial
exact contraction. The fixed-number Pfaffian/AGP subclass is a nontrivial
`chi>1`, strictly antisymmetric, systematically improvable family with exact
polynomial norm and one-/two-body contractions. Phase 4 may proceed only for
this structured family and finite AGP sums.

## Exact validation

- Pfaffian minors equal ordered-channel FEMPS materialization for `M=1,2,3`.
- Fixed-number overlaps equal coefficients of
  `sqrt(det(I+t F^dagger G))`.
- Norm, one-body and factorized two-body values equal full particle tensors.
- Finite sums of distinct AGPs include all bra/ket cross terms and match their
  explicit superposition.
- Reverse-mode gradients with respect to complex pair/channel parameters match
  explicit tensors.
- At `D=64,N=32`, CPU/RTX PRO 4000 Blackwell normalized one-body energy differs
  by `4.09e-12`; the maximum complex-gradient difference is `3.27e-13`.

Raw GPU evidence is in `results/gate_a_agp_gpu_parity.json`.

## Scaling benchmark

CPU complex128, dense pair matrices, median of three runs:

| `D` | `N` | FEMPS bond | Implicit Slater paths | Exterior dimension | Norm (s) | One-body (s) | Norm relative error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 8 | 8 | 70 | 12,870 | 0.00063 | 0.00108 | 4.35e-16 |
| 32 | 16 | 16 | 12,870 | 601,080,390 | 0.00153 | 0.00422 | 3.29e-16 |
| 64 | 32 | 32 | 601,080,390 | 1.83e18 | 0.00594 | 0.00670 | 1.00e-14 |
| 128 | 64 | 64 | 1.83e18 | 2.40e37 | 0.01126 | 0.02746 | 3.42e-14 |

For `D=16,32`, a two-term factorized two-body expectation takes `0.0212 s`
and `0.1154 s`, respectively. Timings are CPU implementation observations, not
asymptotic proofs; the proven recurrence costs are `O(MD^3)` for overlap and
one-body and `O(LMD^3)` for operator-Schmidt rank `L` two-body operators.

Raw scaling data is in `results/gate_a_agp_scaling.json`.

## Stability stress test

The original alternating trace recurrence is not reliable for a generic dense
top sector even when the final answer is representable. At `D=64,N=64` it
returns a negative norm with `2.57e17` relative error. The new positive
singular-value recurrence returns `1.93e-14` relative error, and is about eight
times faster in this single CPU observation (`0.00264 s` versus `0.0218 s`).

The log-norm path remains accurate to `2.28e-13` over pair-matrix scales from
`1e-12` to `1e12`, including cases where the ordinary norm necessarily
underflows or overflows. Separately scaling reciprocal bra/ket inputs by
`1e-120` and `1e120` changes a generic transition overlap by only `3.60e-15`
relative. Raw evidence is in `results/agp_stability_stress.json`.

## Consequence

The original generic FEMPS program is not declared scalable. The admitted
Phase 4 state is now precisely:

1. one fixed-number Pfaffian/AGP state;
2. optionally a finite sum of `K` AGPs;
3. functional-basis one-body matrices and factorized two-body tensors inherited
   from the 2201 pipeline.

The next hard test is whether E1/E2 and then interacting `N=4` converge with
small pair rank/AGP-sum length. Failure there triggers the ordered-sector route
rather than reopening unrestricted exponential matrix-wedge optimization.
