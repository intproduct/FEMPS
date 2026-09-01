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

The Phase 20 `T2` and `Mat2` pair-collapse base cases are checked by:

```powershell
python math/certificates/verify_triangular_pair_collapse.py --verify math/certificates/triangular_pair_lc_agp_certificate.json
python math/certificates/verify_mat2_pair_collapse.py --verify math/certificates/mat2_pair_lc_agp_certificate.json
python -m pytest -q tests/test_triangular_pair.py tests/test_exact_certificates.py
```

The first two commands use exact rational polynomial arithmetic and import
neither PyTorch nor `femps`. The numerical tests independently compare exterior
states, genuinely noncommuting samples, and reverse-mode gradients restricted
to the admitted skew/upper-triangular parameter manifold.

The Phase 21 one-generator growing-radical certificate is checked by:

```powershell
python math/certificates/verify_truncated_polynomial_pair_collapse.py --verify math/certificates/truncated_polynomial_pair_lc_agp_certificate.json
python -m pytest -q tests/test_exact_certificates.py
```

The verifier covers every boundary basis functional for all 16 cases with
`1<=M,d<=4`, so arbitrary boundaries follow by exact linearity rather than a
selected numerical sample.

The Phase 21 alternating-word noncommutative certificate is checked by:

```powershell
python math/certificates/verify_alternating_word_pair_collapse.py --verify math/certificates/alternating_word_pair_lc_agp_certificate.json
python -m pytest -q tests/test_exact_certificates.py
```

It compares direct word-algebra multiplication with nested exact z-coefficient
and `Mat2` power interpolation for every boundary word in all 12 cases with
`1<=M<=3` and `1<=d<=4`.

The Phase 22 sparse-path permanent certificate is checked by:

```powershell
python math/certificates/verify_sparse_path_apg_permanent.py --verify math/certificates/sparse_path_apg_permanent_certificate.json
python -m pytest -q tests/test_exact_certificates.py
```

It compares exact upper-bidiagonal virtual-path propagation, square-zero
commuting exterior subset propagation, and permutation enumeration for three
matrix families at every `1<=M<=6`. The certificate validates the normalized
state and squared-norm convention; the symbolic all-size permanent proof is in
`docs/theory/sparse_path_apg_obstruction.md`.

The current CPU/Blackwell forward-and-gradient parity smoke is:

```powershell
python scripts/gpu_smoke.py
```

The completed continuous ordered-distance Gate D records are reproduced by:

```powershell
python scripts/benchmark_ordered_continuous_controls.py
python scripts/benchmark_ordered_continuous_training.py --device auto
```

The first command uses dense vectors only in bounded truth audits and never
forms the squared product-basis Hamiltonian. The second command is the formal
blind native MPS/MPO path and requires the Blackwell GPU selected by `auto`.

The completed unbounded-interaction Gate E records are reproduced by:

```powershell
python scripts/benchmark_ordered_continuous_fourier.py
python scripts/benchmark_ordered_continuous_fourier_n6.py --device auto
```

The first command includes direct half-line quadrature, compact/direct MPO
equivalence, matched-basis comparisons, and global compression audits. The
second performs every N=6 blind training run before constructing the bounded
product-vector Lanczos and TT-SVD truth audits.

The completed basis-efficiency and N=8 Gate F records are reproduced by:

```powershell
python scripts/benchmark_phase18_basis_and_mpo.py
python scripts/benchmark_phase18_n6_n8.py --device auto
```

The first command checks analytic multiscale-basis operators, matched N=2/N=4
orders, raw-versus-incremental global MPO equality, construction resources, and
a one-million-dimensional N=6 action. The second freezes all blind N=6/N=8
training choices before same-basis truth/reference audits. It writes an ignored
recoverable checkpoint before local DMRG, records the chi-32 resource rejection,
and distinguishes the Gate F core pass from the failed auxiliary raw-gradient
threshold.

The completed Phase 19 Gate G evidence is reproduced from fresh AD states by:

```powershell
python scripts/benchmark_phase19_dmrg_memory.py --device auto
python scripts/benchmark_phase19_mpo_tangent.py --device auto
python scripts/benchmark_phase19_mpo_bond_training.py --device auto
python scripts/benchmark_phase19_n8_d12_basis.py --device auto
python scripts/build_phase19_resource_trend.py
```

These respectively audit the staged chi-32 effective-Hamiltonian contraction,
left-gauge physical tangent derivatives across MPO bonds, matched bond
training, blind D12 basis/operator/reference refinement, and the descriptive
N=2/4/6/8 accuracy/resource trend. The raw Gate F tensor-gradient miss remains
in its original record. The D14 exterior value is a numerical reference rather
than a continuum certificate, and the trend explicitly does not admit N=10.
