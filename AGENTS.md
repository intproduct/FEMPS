# FEMPS agent guide

## Three red lines

1. Never silently break antisymmetry; every approximation must report an
   antisymmetry residual.
2. Never present exploratory floating-point numerics as a theorem or proof.
3. Never relabel an occupation-number/second-quantized MPS as FEMPS. The main
   method remains first-quantized and uses continuous particle coordinates.

## Documentation entry points

- Authoritative scientific plan: `AGENT.md`
- Architecture: `docs/ARCHITECTURE.md`
- Evidence status: `docs/THEORY_STATUS.md`
- Testing: `docs/TESTING.md`
- Active execution plan: `docs/exec-plans/active/phase25.md`
- Research log: `CHANGELOG.md`

## Standard validation

```powershell
python -m pytest -q
python scripts/reproduce_2201_baseline.py --device cpu
```

Tests are CPU-only by default and use float64/complex128 references. New
contraction primitives require small-system materialization equivalence and AD
gradient checks before performance work.

## Evidence labels

Use exactly these labels in research documents: **theorem**, **exact
certificate**, **numerical evidence**, and **conjecture**. Update the active
plan and `CHANGELOG.md` after each accepted task. The Master Plan governs any
conflict.
