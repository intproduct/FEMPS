# arXiv:2201.12823 functional-MPS baseline

## Reproduction target

The first target covers Eqs. (9)-(11), (22), (27), (29), and (31) and the
two-body/no-three-body data underlying Figs. 3 and 4:

- single-oscillator harmonic basis of order `D`;
- analytic position and derivative matrices;
- `N` oscillators with unit bare frequency and nearest-neighbor coupling;
- MPS coefficient tensor with bond dimension `chi`;
- differentiable Rayleigh quotient optimized with Adam;
- deterministic cosine learning-rate decay to prevent late-time oscillation;
- post-step per-core scalar normalization, which leaves the represented
  Rayleigh state unchanged and matches the latticeTN stabilization policy;
- comparison to the exact normal-mode ground energy.

The long three-body scan is deferred until the two-body baseline is stable.

## Parameter discrepancy recorded from the paper

The opening sentence of Sec. IV says `gamma = 0.5`, while the Fig. 3 and Fig. 4
captions say `gamma = -0.5`. Equation (31) is invariant under the sign of
`gamma` for the complete open-chain mode sum, but wavefunctions and individual
mode ordering are not identical. The reproduction config therefore always
records the sign explicitly and defaults to `-0.5`, matching the captions.

## Operator convention correction

For column index `s`, the mathematically consistent harmonic derivative is

```text
D[s-1, s] = sqrt(s/2)
D[s+1, s] = -sqrt((s+1)/2)
```

The rendered Eq. (9) appears to use `sqrt((s+1)/2)` in both branches. Tests use
the ladder-operator identity and treat the first branch as a typographical
error. Boundary identities are checked only away from the truncated top state.

## Acceptance checks

1. `X` is Hermitian and `D` is anti-Hermitian in the truncated basis.
2. Low-lying single-oscillator energies are `n + 1/2` to float64 precision.
3. The latticeTN native functional energy equals a dense Rayleigh quotient for
   a small random MPS.
4. AD lowers the energy and never crosses below the exact continuum energy by
   more than the declared truncation/numerical tolerance.
5. Every run emits configuration, environment, seed, initial/final energy, and
   error as JSON.
