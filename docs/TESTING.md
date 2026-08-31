# Testing strategy

The validation hierarchy follows the Master Plan.

1. **L0 unit tests:** shapes, dtype/device, harmonic operators, serialization.
2. **L1 algebraic properties:** wedge signs, repeated-vector zero,
   associativity, and permutation signs.
3. **L2 materialization equivalence:** every new small FEMPS contraction equals
   an explicitly antisymmetrized coefficient tensor.
4. **L3 exact physics:** oscillator energies, Slater overlaps, and exact
   diagonalization.
5. **L4 AD:** autograd versus finite difference and CPU/GPU parity.
6. **L5 regression:** fixed seed, energy/norm/gradient/rank outputs.
7. **L6 certificates:** exact arithmetic, field metadata, hashes, and an
   independent verifier.

Default tests are deterministic, CPU-only, and use float64 or complex128.
GPU benchmarks are opt-in and are never hidden inside pytest.

The current CPU/Blackwell forward-and-gradient parity smoke is:

```powershell
python scripts/gpu_smoke.py
```
