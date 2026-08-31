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

See [the active soft-Coulomb hierarchy plan](docs/exec-plans/active/phase10.md),
[the Pfaffian theory](docs/theory/pfaffian_subclass.md), and
[the 2201 reproduction report](docs/experiments/2201_baseline_report.md).

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

- `src/femps/basis`: continuous functional bases and operator matrices;
- `src/femps/baselines`: the differentiable 2201 MPS baseline;
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
