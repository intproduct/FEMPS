# Research changelog

## 2026-08-31 - Phase 0 bootstrap

- Established the Python package, CI, documentation, reference-management, and
  test skeleton.
- Pinned the development integration point for `latticeTN` at commit
  `9d4c857270a310af24a7133c32275cb79f800c9f`.
- Defined the first arXiv:2201.12823 reproduction target: harmonic functional
  operators and the no-three-body coupled-oscillator energy baseline.
- Added a continuum-safe MPS initializer because the upstream `MPS`
  constructor caps bonds using a local dimension of two.
- Installed and validated PyTorch 2.11.0+cu128. The workstation enumerates the
  RTX PRO 4000 Blackwell as `cuda:2`; automatic selection now avoids the two
  earlier V100 devices, which are unsupported by this wheel's architecture set.
- Reproduced the `N=4, D=8, chi=16, gamma=-0.5` two-body functional-MPS
  baseline at `1.16e-5` absolute energy error on the Blackwell GPU.
- Completed a 14-point controlled `D`, `chi`, and seed scan. Basis and bond
  convergence are variational, while the four-seed anchor spread identifies a
  finite-optimization floor near `1e-5`.
- Refactored the baseline optimizer into a reusable, validated training API and
  added resumable scan tooling with machine-readable summaries.
- Completed the first representation audit of Li-Waintal ordered-sector MPS,
  Li-Chan HS-MPS, and the 2026 Grassmann tensor-network review.
- Added the ordinary particle-TT no-go proof draft: exact unfolding ranks,
  universal binomial rank floor, flat Slater particle-Schmidt spectrum, and the
  corresponding approximate-bond lower bound.
- Added an exponential small-system exterior reference engine with independent
  antisymmetrizer/minor constructions and strict antisymmetry diagnostics.
- Fixed the normalized exterior-Hilbert-space convention and formal matrix-wedge
  FEMPS definition, including associativity, strict antisymmetry, the `chi=1`
  theorem, finite Slater-sum embedding, ordinary gauge action, and exact `N=2`
  bond characterization.
- Added two independent explicit FEMPS materializers and cross-checks for
  `N=2,3,4`; both remain exponential truth oracles pending Gate A.
- Added three independent exact norm routes (full tensor, determinant paths,
  exterior-coordinate dynamic programming), generalized one-body cofactors,
  and a minimal two-body cofactor implementation.
- Derived explicit Gate A costs. Exterior propagation removes exponential
  virtual-path enumeration but retains `binom(D,p)` state dimension; generic
  polynomial contraction therefore remains open.
- Benchmarked exact norm scaling in both `N` and `chi`. The exterior recurrence
  changes exponential path-pair growth into low-order bond growth, but its
  central exterior sector displays the predicted combinatorial memory cost.
- Cross-checked fermionic operator-circuit and Gaussian/matchgate contraction
  literature. Graded signs alone do not lower contraction width; polynomial
  closure points instead to a restricted Gaussian/Pfaffian fallback.
- Added pinned CPU, CUDA 12.8, and exact upstream `latticeTN` dependency files,
  and validated the installable FEMPS wheel.
- Issued a CONDITIONAL Gate A decision for a fixed-number Pfaffian/AGP FEMPS
  subclass. Unrestricted matrix-wedge contraction remains unapproved.
- Implemented Pfaffian minors, ordered-channel FEMPS embedding, polynomial
  overlap/one-body recurrence, factorized two-body derivatives, and finite AGP
  sums with full complex AD support.
- Verified the structured contractions against explicit tensors and on RTX PRO
  4000 Blackwell. A `D=128,N=64` norm contracts in about `0.011 s` on CPU while
  representing `1.83e18` ordered Slater paths.
- Connected the Pfaffian engine to harmonic functional operators and completed
  E1/E2 AD benchmarks on Blackwell. E1 is exact at energy `2`; interacting E2
  reaches the finite-basis truth within `8.9e-16` and the continuum value within
  `4.6e-12` at `D=12`.
- Added deterministic checkpoint/resume and corrected the projected `x^2`
  matrix at the truncated top boundary.
- Completed the E2 basis scan from `D=4` through `D=14`, with monotone
  continuum error reduction from `1.01e-3` to `2.93e-14`.
- Added a constrained real-skew pair-channel parameterization and canonical
  decomposition oracle. Three channels reach `7.3e-13` error against the
  `D=12` antisymmetric truth, while the ordinary particle-TT rank is twelve.
- Restored the best recorded factorized-pair iterate and retained the terminal
  Adam energy in the raw record, documenting rather than masking the nonconvex
  factor-gauge drift.
- Replaced self-norm Newton traces with a positive paired-singular-value
  recurrence, added an overflow-safe log norm, and homogeneously scaled generic
  transition overlaps. A dense `D=64,N=64` stress case improves from a wrong
  sign and `2.57e17` relative error to `1.93e-14`.
- Added single-orbital blocked Pfaffian FEMPS for odd particle number, including
  polynomial overlap and one-/two-body contractions by auxiliary-sector
  subtraction. Exact `D=5,N=3` truth checks and `D=32,N=21` Blackwell gradient
  parity pass.
- Added a general small-sector Slater--Condon Hamiltonian oracle and completed
  E3. Four noninteracting fermions have energy `8`, ordinary particle-TT ranks
  `(1,4,6,4,1)`, flat Schmidt spectra, and direct FEMPS correlation bond one;
  blind Blackwell AD reaches `1.87e-14` energy error.
- Completed E4 basis, coupling, seed, and finite-AGP-length scans. At
  `D=8,kappa=0.35`, the oracle representation error falls from `3.11e-3` for
  one AGP to `2.81e-9` for eight, with polynomial/exterior energy agreement at
  `1.8e-14`; random `K=2` optimization exposes a remaining nonconvex solver
  bottleneck.
- Added finite-AGP overlap/Hamiltonian transition matrices and a conditioned
  generalized Hermitian amplitude solver with overlap-rank, condition-number,
  and residual diagnostics. Duplicate AGP directions are removed exactly in
  regression tests.
- Added resumable variable-projection training with pair scale/phase gauges and
  deterministic output ordering. Greedy no-oracle K=1-to-K=2 growth produces
  three reproducible E4 errors of `2.00e-5`--`3.11e-5`, around two orders of
  magnitude below simultaneous random K=2 training.
- Completed E5 at six particles. The noninteracting Slater has ordinary ranks
  `(1,6,15,20,15,6,1)` versus FEMPS correlation bond one; at
  `D=10,kappa=0.1`, greedy K=2 reaches `4.765e-6` finite-basis error while the
  explicit ordinary tensor has full internal ranks `(10,45,120,45,10)`.
