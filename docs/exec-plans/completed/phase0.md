# Completed execution plan: Phase 0

## Objective

Establish a clean FEMPS repository, integrate the pinned sibling `latticeTN`,
and reproduce the functional-basis MPS baseline of arXiv:2201.12823 before
starting exterior-algebra solver work.

## Checkpoints

- [x] Repository, package, CI, docs, and reference skeleton.
- [x] Harmonic-oscillator `X`, `d/dx`, kinetic, and one-body Hamiltonian
  matrices.
- [x] Coupled-oscillator exact normal-mode energy and differentiable MPS energy.
- [x] Clean environment installation and CPU test pass.
- [x] CUDA detection and CPU/GPU contraction parity on RTX PRO 4000 Blackwell.
- [x] Reproduce a small stable subset of Figs. 3/4 with raw JSON output.
- [x] Record the exact `D` and `chi` convergence table, including a four-seed
  stability check and per-run raw JSON.

The exact small-`N` antisymmetric reference harness was completed jointly with
Phase 1's numerical theorem checks. All Phase 0 items and the exit criterion
are satisfied.

## Exit criterion

A clean checkout runs unit tests with one command, and an AD optimization using
functional operator matrices reaches a documented coupled-oscillator reference
without materializing the full coefficient tensor.
