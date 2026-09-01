# FEMPS

FEMPS (working name: **Functional Exterior Matrix Product State**) is a
first-quantized research project for continuous fermionic Schrödinger
problems. It extends the functional tensor-network construction of Hong et
al. (2022) while treating antisymmetry as an exterior-algebra structural layer
rather than forcing it into an ordinary particle-site tensor train.

The project is deliberately staged. The functional-basis MPS baseline of
arXiv:2201.12823 has been reproduced first. Exterior/FEMPS algorithms are
admitted only after small-system exact tests, and scalable solver development
is gated by an explicit contraction-complexity audit.

## Current status

- A controlled `D`, `chi`, and seed scan reproduces the 2201 coupled-oscillator
  baseline at the `1e-5` energy-error scale on RTX PRO 4000 Blackwell.
- The ordinary particle-TT no-go theorem draft and executable small-system
  antisymmetric reference engine are complete.
- Matrix-wedge FEMPS is formally defined and cross-checked for `N=2,3,4`,
  including `chi=1`, finite Slater sums, gauge invariance, and the exact
  two-particle characterization.
- Gate A is CONDITIONAL: generic exterior propagation remains combinatorial,
  while fixed-number Pfaffian/AGP FEMPS and finite sums have exact polynomial
  norm and functional-operator contractions.
- E1/E2 reach their finite-basis truths, stable/log contractions and odd blocked
  Pfaffians pass exact/GPU checks, and E3 demonstrates ordinary ranks
  `(1,4,6,4,1)` versus correlation bond one at energy `E=8`.
- E4/E5 add conditioned finite-AGP variable projection and greedy growth. Six
  noninteracting fermions have ordinary ranks `(1,6,15,20,15,6,1)` versus
  FEMPS correlation bond one; the interacting `N=6,D=10,K=2` benchmark reaches
  `4.765e-6` finite-basis error with independent exterior truth.
- Phase 8 reaches `N=8` without materializing the `10^8`-entry particle tensor
  and adds an exact ordered-coordinate hard-wall oracle as an independent
  competing-representation control.
- E6 introduces a converged factorized soft-Coulomb operator. Blind/restarted
  N=2 reaches finite-basis truth, N=4 greedy K=2 reaches `7.445e-5` error, and
  batched mixed-derivative contractions remove the initial factor-axis bottleneck.
- The N=4 soft-Coulomb hierarchy now reaches K=4 with three-seed reproduction;
  independent exterior truth extends through D=14, while D=10 conditioning
  identifies canonical finite-AGP compression as the next solver task.
- Generic matrix-wedge FEMPS exact contraction is now conditionally obstructed
  by a polynomial tagged reduction from the permanent-hard `Mat_2` Cayley
  determinant; tractable restricted cases found so far collapse to established
  AGP/Gaussian structure.
- Gate C passes for the finite-grid ordered-distance route. Exact hard-charge
  gap MPS/MPO contraction has raw operator bond `O(N^2(L-N))`; three blind
  `N=4,L=8` Blackwell runs reproduce the truth within `2.07e-5` without a
  `d**(N+1)` training gather.
- Gate D passes at controlled `N<=4` scope for the continuous bridge. Exact
  COM/positive-gap coordinates, Dirichlet and odd-Hermite half-line bases,
  continuum mixed-derivative/soft-Coulomb MPOs, and native latticeTN AD now
  have independently controlled `D`, scale/box, interaction-degree, MPS-bond,
  and optimization errors. Three blind N=4 runs are within `4.39e-3` of an
  independent numerical reference.
- Gate E passes at controlled `N<=6` scope. A Fourier--Bessel soft-Coulomb
  separation supports the unbounded odd-Hermite basis, and a four-state real
  recurrence per Fourier node gives all-pair interaction bond `4M`, independent
  of particle count. Three blind N=6 Blackwell runs are within `2.49e-4` of
  their post-run Galerkin truth; the production MPO compression has `1.63e-9`
  global action error. The remaining `1.33e-2` total discrepancy is basis
  dominated, so basis efficiency and N=8 define the next gate.
- Gate G passes at a resource-closed controlled `N=8,D=12` point. A staged
  chi-32 local effective-Hamiltonian action peaks below 1.1 GB, bond 128 passes
  a left-gauge physical-tangent audit against bond 192, and matched bond
  training retains 128 as the smallest passing production value. A blind D12
  multiscale run lies `7.174e-3` above an exterior D14 numerical reference,
  improving the Gate F D10 error against that same reference by `17.6%`.
  D14 remains finite-basis numerical evidence; the N=2/4/6/8 resource trend
  does not admit N=10 or an asymptotic scaling claim.
- Phase 20 closes Gate H negatively for uniformly bounded coefficient
  algebras. If both the largest semisimple matrix block and radical nilpotency
  index are fixed, every arbitrary-boundary matrix-pair power collapses to a
  polynomial-size exact LC-AGP. Independent exact-rational certificates pass
  for upper-triangular `T2` at M=1--6 and fully noncommutative `Mat2` at M=1--4.
  A genuinely beyond-LC-AGP exact family must therefore introduce growing
  semisimple blocks or growing radical memory together with new contractible
  structure.
- Phase 21's first growing-memory candidate also collapses. For
  `C[z]/(z^d)`, arbitrary boundaries need at most `M(d-1)+1` scalar AGPs, with
  all boundary basis functionals certified exactly for `1<=M,d<=4`. A single
  commuting path coordinate is therefore insufficient even when its memory
  depth grows; the next exact candidate must be genuinely multibranch and
  noncommutative.
- Phase 21 then rejects the minimal noncommutative alternating-word algebra and
  its fixed-state graded generalization. The `2d-1` dimensional algebra embeds
  in `Mat2(C[z]/z^d)` and needs at most
  `[M(d-1)+1] binom(M+3,3)` AGPs; every boundary word is certified exactly for
  `1<=M<=3,1<=d<=4`. Gate I closes negatively.
- Phase 22 closes the sparse growing-width path gate. The weakest upper-
  bidiagonal endpoint state is exactly APG/APIG. With paired edge forms it
  encodes an arbitrary 0--1 permanent in a unique-path top coefficient and
  `perm(A)^2/(M!)^2` in the normalized exact squared norm, despite bandwidth one and
  only `O(M^2)` input. Gate J therefore fails on both direct-prior-art and
  exact-contraction grounds. The result is not an LC-AGP rank lower bound.
- Phase 23 consolidates the result into a two-axis no-go hierarchy. The sparse
  APG permanent is now the simplest generic exact squared-norm proof; the
  tagged Cayley construction independently diagnoses growing row-order memory.
  Fixed `Mat2` pair powers are explicitly corrected to the polynomial LC-AGP
  side. The conclusion covers the tested exact coefficient-memory corridor,
  not every structured or approximate exterior method.
- Phase 27 establishes the first two exact values in the independent
  four-form program. Source-complete orbit classifications and independent
  exact certificates give `mu_4^Q(7)=mu_4^Qbar(7)=12` and
  `mu_4^C(8)=mu_4^Qbar(8)=mu_4^Q(8)=12`. The 16D rank-22/23 provenance is
  still missing; the only certified 16D statement remains the rational upper
  bound `mu_4^Q(16)<=24`.
- Phase 27 is now parked. Phase 28 restores the solver/physics main line with a
  restricted nonbranching diagonal-path FEMPS: `K` globally conserved Slater
  paths contract through `K^2` determinant transitions. The first E1/E2
  artifact verifies energies and gradients against full exterior truth and
  shows numerical `K=1,2,4` convergence for an interacting two-fermion model.
  This route is close to nonorthogonal selected CI and carries no novelty or
  scalability claim before the full physics ladder and comparator audit pass.

See [the active execution plan](docs/exec-plans/active/phase28.md),
[the Phase 28 physics-ladder report](docs/experiments/phase28_diagonal_path_ladder_report.md),
[the exact four-form workspace](math/four_forms/README.md),
[the no-go hierarchy](docs/theory/exterior_no_go_hierarchy.md),
[the continuous ordered formulation](docs/theory/continuous_ordered_functional_basis.md),
[the Gate G report](docs/experiments/phase19_n8_refinement_gate_report.md),
[the Phase 20 negative-classification report](docs/experiments/phase20_bounded_wedderburn_report.md), and
[the Phase 22 sparse-path report](docs/experiments/phase22_sparse_path_gate_report.md).

## Development setup

FEMPS and `latticeTN` are sibling repositories during development:

```text
workspace/
|-- FEMPS/
`-- latticeTN/
```

Create a virtual environment, install the pinned CUDA environment when a
Blackwell GPU is available, and install both packages in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements\gpu-cu128.txt
.\.venv\Scripts\python -m pip install -e ..\latticeTN -e .
```

For a non-editable checkout, `requirements/latticetn.txt` records the exact
upstream Git commit. CPU-only CI uses the pinned CPU environment and excludes
the sibling-integration test.

## Validation

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python scripts\reproduce_2201_baseline.py --device cpu
```

GPU use is opt-in and never required by default tests:

```powershell
.\.venv\Scripts\python scripts\reproduce_2201_baseline.py --device auto
```

On the current workstation, PyTorch enumerates the two V100 cards before the
RTX PRO 4000 Blackwell. `auto` selects the compatible GPU with the highest
compute capability rather than assuming that `cuda:0` is the Blackwell card.

## Repository map

- `src/femps/basis`: full-line and half-line functional bases and operators;
- `src/femps/baselines`: the differentiable 2201 MPS baseline;
- `src/femps/ordered_distance.py`: exact finite-box ordered gap coordinates;
- `src/femps/baselines/ordered_distance_mpo.py`: native gap MPS/MPO operators;
- `src/femps/ordered_continuous.py`: exact COM/positive-gap continuum map;
- `src/femps/baselines/ordered_continuous_mpo.py`: continuous functional MPOs;
- `src/femps/baselines/ordered_continuous_fourier.py`: unbounded
  Fourier--Bessel interaction, compact all-pair recurrence, and incremental
  structured compression;
- `src/femps/basis/multiscale_odd_hermite.py`: collision-compatible two-scale
  half-line basis and analytic projected local operators;
- `src/femps/exterior`: exact antisymmetric, matrix-wedge, and Gate A oracles;
- `docs/theory`: theorem and contraction-status drafts;
- `docs/experiments/results`: machine-readable reproduction records;
- `math`: standalone LaTeX theorem drafts;
- `references`: bibliography, novelty matrix, and reading notes.

## Scientific guardrails

1. No silent loss of fermionic antisymmetry.
2. No floating-point experiment is presented as a proof.
3. No drift from a first-quantized continuous functional representation into
   an occupation-number MPS under the FEMPS name.
4. No polynomial-contraction claim based only on polynomial parameter count.

The authoritative project plan is [AGENT.md](AGENT.md).
