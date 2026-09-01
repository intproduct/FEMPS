# FEMPS method claim and evidence matrix

This matrix governs method-facing language after Phase 29. Machine-readable
artifact hashes, commands, seeds, tolerances, boundaries, and verifier results
are in `docs/experiments/results/phase30_reproduction_manifest.json`.

| Claim | Evidence status | Authoritative evidence | Allowed wording |
|---|---|---|---|
| Matrix-wedge/exterior composition is exactly antisymmetric | proved structure plus exact small-system tests | theory status; exterior and diagonal-path unit tests | exact antisymmetry by construction |
| Generic matrix-wedge FEMPS has polynomial exact squared-norm contraction | conditionally obstructed at fixed small bond | no-go theorem draft and exact certificates | generic exact contraction is not assumed |
| Nonbranching diagonal-path FEMPS contracts through K-squared determinant transitions | implemented restricted algorithm; value/gradient verified | solver contract, transition tests, E4 and matched-cost artifacts | polynomial exact contraction for this restricted subclass |
| Single Slater has FEMPS correlation multiplicity one while ordinary particle TT carries exchange ranks | proved/numerically verified | Slater spectrum theorem/tests and comparator artifacts | exchange carrier is structurally separate from K in this representation |
| Increasing K recovers non-Slater correlation | numerical evidence at N2/N4/N6 | E4, high-basis K4-to-K5, and N6 K1-to-K4 artifacts | systematic improvement at the tested points |
| Increasing D gives controlled functional-basis convergence | numerical evidence at N2/N4/N6 | E4, N4 D8-to-D12, and N6 D8/D10/D12 artifacts | monotone absolute energy on registered axes; no asymptotic or continuum bound |
| Restricted FEMPS solves a nontrivial continuous interacting model reproducibly | numerical evidence | N4 soft-Coulomb multiseed and N6 three-seed artifacts | stable optimization for the verified N4/N6 benchmarks |
| Antisymmetry survives every approximation/validation step | exact structural residual plus bounded materialized audits | all registered artifacts and manifest verifier | residual is always reported; materialized residual is zero where enabled |
| FEMPS is faster than CI/DMRG | not established | CI is faster in current truth spaces; DMRG not yet admitted | no runtime-superiority claim |
| FEMPS scales to N8 or asymptotically | not established | no admitted N8 diagonal-path artifact | no N8 or asymptotic claim |
| Batched transition contractions accelerate the reference implementation | numerical backend evidence at N6,D10,K4 | Phase 33 value/gradient parity and matched CPU/Blackwell artifact | CPU batched kernel is faster than the pairwise reference; Blackwell is admitted but CPU remains faster on the registered workload |
| Truth-free adaptive K growth improves a fixed interacting N6 state | numerical evidence for one D12 seed-pool lineage | Phase 34 K4-to-K5-to-K6 artifact, cold K6 control, and independent exterior verifier | measurable monotone improvement for the registered lineage; no asymptotic or superiority claim |
| Adaptive K growth is stable across fresh candidate pools | numerical evidence for three new D12 K5/K6 lineages | Phase 35 six-optimization artifact and independent exterior verifier | all three lineages improve and have `K=6` energy spread `4.877e-6`; automatic stopping remains unadmitted because no stop event occurred |
| Adaptive K growth is available as a reproducible public solver operation | implemented API plus numerical reproduction | Phase 36 schemas, identity/resume/materialization/AD tests, physical artifact, and independent verifier | bounded explicit K schedules are checkpointable and reproduce one frozen N6,D12 lineage; no automatic-stop or generic-solver claim |
| A clean canonical Slater can drive the public solver without historical FEMPS state | implemented command plus numerical reproduction | Phase 37 command/checkpoint schemas, N2 materialization/AD/resume tests, N4,D6 artifact, and independent verifier | K1--K4 clean and resumed runs agree exactly and reach `4.883e-10` same-basis CI error; one bounded truth point, no scaling or automatic-stop claim |
| Diagonal-path FEMPS is novel beyond nonorthogonal selected CI | not established | novelty comparison remains qualified | describe the exterior/functional integration and measured tradeoff, not categorical novelty |

## Required paper language

- Every floating-point table or figure is labeled numerical evidence.
- `K` is a restricted determinant-path correlation multiplicity, not a
  canonical entanglement spectrum.
- Dense CI, Slater, AGP, ordinary particle TT, ordered-sector methods, and any
  future second-quantized DMRG calculation retain their own names.
- The current practical claim is bounded to the verified nonbranching
  first-quantized continuous functional-basis subclass.
