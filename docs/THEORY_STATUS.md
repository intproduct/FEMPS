# Theory and evidence status

| Claim | Status | Required next evidence |
|---|---|---|
| A decomposable normalized Slater state has a flat particle-cut Schmidt spectrum | proved in draft and numerically verified | External proof review and literature cross-check |
| Ordinary particle TT must pay the exchange-statistics Schmidt multiplicity | proved in draft and numerically verified | Stronger concise/full-support bounds remain open |
| The seven-dimensional concise four-form minimum is `mu_4^Q(7)=mu_4^Qbar(7)=12` | **theorem draft**: Cohen--Helminck supplies exhaustive nine-orbit coverage after scalar extension; an implementation-independent exact certificate verifies every orbit rank, conciseness flag, and payload hash | External algebra review; audit the characteristic-positive and real-field refinements separately before extending the field statement |
| The eight-dimensional concise four-form minimum is `mu_4^C(8)=mu_4^Qbar(8)=mu_4^Q(8)=12` | **theorem draft**: Antonyan--Oeding supplies the Cartan subspace and all 94 nilpotent orbits; theta-group closure reduces an arbitrary low-rank candidate to those two cases; an independent exact certificate verifies the Cartan eigenbasis, `F_3` hyperplane bound, every nilpotent rank, and both payload hashes | External invariant-theory review; audit the real and positive-characteristic variants separately before extending the field statement |
| The 16D concise four-form minimum middle rank is 22 or 23 | **conjecture** with missing provenance; neither alternative is currently admitted. An **exact certificate** gives only the explicit rational upper bound `mu_4^Q(16)<=24`; a 120,000-sample finite-field screen found no lower coordinate-hypergraph candidate | Recover a primary definition/candidate or construct one from scratch; certify rank and complete orbit/chart coverage over an explicit field before claiming sharpness |
| Matrix-wedge FEMPS is associative and strictly antisymmetric | proved from exterior associativity; small-N verified | External algebra review |
| Generic nontrivial FEMPS has polynomial exact contraction | conditionally obstructed at fixed maximum bond three: site-indexed one-form bond-two cores encode the `Mat_2` row-ordered Cayley determinant pointwise, while the scalar-reference direct sum used to recover signed amplitudes from exact squared norms raises the maximum bond to three; `chi=2` exact-norm hardness is a conjecture, not a theorem; the Phase 13 tagged pair-power and Phase 22 bandwidth-one APG reductions remain independent mechanisms | External proof review; only separately gated restrictions or controlled approximation remain open |
| Nonbranching diagonal-path FEMPS has polynomial exact contraction | **implemented NOCI-equivalent restricted algorithm / numerical physics evidence**: a conserved global path label gives a `K`-term nonorthogonal Slater sum inside matrix-wedge FEMPS, eliminating `K^(N-1)` branching; E1--E4, N4 convergence, N6 stability and independent D/K convergence pass; Phase 33 batches transitions; Phase 34 truth-free adaptive D12 growth lowers the K4 CI error from `1.047e-4` to `3.202e-5` at K6 and beats a same-budget cold K6 control; Phase 35 passes three fresh K5/K6 candidate-pool lineages with K6 energy spread `4.877e-6`; Phase 36 exposes the externally capped schedule as a versioned public API; Phase 37 starts from a canonical Slater with no historical FEMPS state and reaches N4,D6,K4 CI error `4.883e-10`; Phase 38 passes two fresh clean-source schedules with distinct candidate paths, combined final-energy spread `2.035e-9`, maximum CI error `2.523e-9`, and zero failures | Retain as a numerical chapter in the combined paper. It cannot support a separate method paper without a non-NOCI explicit-correlation D-convergence advantage or a matched Li--Waintal/same-basis-DMRG tradeoff; automatic stopping, novelty, scaling, and superiority remain unadmitted |
| Generic sparse-path FEMPS has a relative squared-norm PRAS | conditionally obstructed: on real-PSD paired coefficients, `M! sqrt(n_tilde)` would be a PRAS for the PSD permanent, which Meiburg excludes unless `RP=NP`; an exact-rational conditioning/energy certificate passes | External proof review; entrywise-nonnegative FPRAS, additive estimates with a certified norm lower bound, and separately promised structures remain open |
| Every exterior cut support factors as fixed statistics carrier times canonical correlation multiplicity, with Slater multiplicity one | disproved for the direct tensor product: Slater `r_1=N` forces carrier dimension `N`, but a full-support two-Slater family has stable `r_1=N+2`; exact all-cut controls pass for `3<=N<=8` | External proof review; Hamiltonian-specific symmetry sectors and genuinely different categorical structures are not ruled out |
| Uniformly bounded coefficient algebras define a distinct tractable matrix-pair family beyond finite LC-AGP | disproved in theorem draft: if the largest semisimple matrix block `p` and radical nilpotency index `d` are fixed, arbitrary-boundary pair powers collapse to polynomial-size LC-AGP; exact-rational `T2` M=1--6 and `Mat2` M=1--4 base cases pass | External proof review; search only growing-`p`/growing-`d` exact structures or controlled approximate contractions |
| One-generator growing radical `C[z]/(z^d)` escapes polynomial LC-AGP | disproved in theorem draft: arbitrary boundaries use at most `M(d-1)+1` AGPs; all boundary basis functionals pass exact-rational checks for `1<=M,d<=4` | External proof review; classify the smallest multibranch noncommutative growing-memory algebra |
| Fixed-state noncommutative graded memory escapes polynomial LC-AGP | disproved in theorem draft: fixed matrix width `w` and fixed commuting-counter count `g` give an explicit polynomial LC-AGP bound; the `2d-1` dimensional alternating-word algebra passes every exact boundary at `1<=M<=3,1<=d<=4` | External proof review; incorporated into the Phase 23 no-go synthesis |
| Growing-width fixed-bandwidth path matrices give a new exactly contractible exterior family | rejected: the upper-bidiagonal unique-path state is established APG/APIG, and paired edge forms encode a 0--1 permanent in the normalized exact squared norm using `D=2M`, `w=M+1`, and `O(M^2)` input; three independent exact routes pass for `1<=M<=6` | External proof review and manuscript-level integration; do not infer an LC-AGP rank lower bound from this contraction reduction |
| Fixed-number Pfaffian/AGP FEMPS contracts polynomially | theorem/algorithm draft; value, gradient, scaling and GPU verified; the LC-AGP family and K-squared organization have direct prior art | Use as fallback/control, not central novelty |
| Finite-AGP overlap compression can be made term-gauge invariant | proved by unit-diagonal balancing; explicit exterior and three-seed D10 tests pass | Relation of the contribution Gram spectrum to physical correlation remains open |
| 2201 functional MPS baseline transfers to the current backend | validated by controlled `D`/`chi`/seed scan | Larger paper-figure digitization is optional |
| Ordered-distance MPS/MPO has polynomial native contraction on the finite grid | Gate C pass: exact gap bijection, hard charge, kinetic/trap/soft-Coulomb MPO with raw bond `O(N^2(L-N))`; three blind N4,L8 runs reach `1.30e-6`--`2.07e-5` energy error | Continuous bridge is addressed by Gate D; finite-grid scaling remains an independent control |
| Continuous ordered-distance functional MPS/MPO is a controlled 2201 bridge | Gate G pass at a resource-closed controlled N=8,D=12 point: staged chi-32 local action below 1.1 GB, gauge-fixed physical-tangent MPO audit, matched bond training, D14 exterior numerical reference, and independently controlled basis/Fourier/quadrature errors | Retain N=10 and asymptotic/continuum claims as unadmitted; use this branch as a control while returning to a genuinely distinct tractable exterior structure |

Scalable contraction is asserted only for the fixed-number Pfaffian/AGP
subclass and finite sums, not for generic matrix-wedge FEMPS. No novelty claim
is currently asserted as fact.  The finite-AGP contribution Gram spectrum is a
gauge-invariant multiplicity diagnostic, not an entanglement spectrum. Phase 12
shows that the LC-AGP subclass is numerically systematic, but prior AGP-CI work
precludes treating that fact as the core FEMPS method contribution.

Phase 26 strengthens the generic status from joint-polynomial hardness at
growing bond to fixed-parameter hardness: exact squared-norm contraction for
the admitted one-form class with `chi<=3` would imply `FP=#P`. The direct
construction preserves row order through site labels and therefore does not
contradict the polynomial LC-AGP collapse of a repeated homogeneous `Mat_2`
pair power. This does not apply automatically to additional structured core
algebras or approximation. Phase 24 separately
rules out a generic **relative** squared-norm PRAS through real-PSD permanent
inapproximability, while preserving entrywise-nonnegative and additive/
conditioned approximation as explicit special cases. Phase 25 then tested the
remaining statistics-carrier/correlation-multiplicity factorization directly.

Phase 25 rejects that factorization in its literal direct-tensor-product form.
The rank obstruction is coordinate/gauge invariant and stable, while standard
symmetry adaptation yields either the one-dimensional sign irrep or the full
orbital exterior representation. The Phase 22 projective-Slater output also
shows that intrinsic multiplicity one does not evaluate a hard scalar from
compact cores. Phase 26 consolidates the full result package in a single
proof-audited manuscript. Its exact-contraction obstruction is now interpreted
as an algorithm-design constraint: it rules out a generic exact contraction
engine for the admitted class, but it does not rule out structurally restricted,
additive/conditioned stochastic, or otherwise controlled approximate FEMPS
algorithms.

Phase 27 reconstructed a source-backed contraction-rank convention and
an explicitly provisional definition of `mu_4^K(m)`. Repository and Git-history
search found no inherited 16D candidate or certificate. The exact rational
direct-sum control has Hilbert/rank vector `(1,16,24,16,1)` and is independently
verified, but it is only an upper bound. The first seeded sparse screen is
**numerical evidence** and cannot exclude non-coordinate, special-coefficient,
or unsampled characteristic-zero orbits. Separately, exterior apolarity now
identifies every rank vector with the Hilbert vector of a canonical exterior
Poincare-duality quotient. Cohen--Helminck's complete seven-dimensional orbit
table plus independent rational reranking closes
`mu_4^Q(7)=mu_4^Qbar(7)=12`. Antonyan--Oeding's Cartan and 94 nilpotent
normal forms, joined by the theta-group orbit-closure theorem and independently
verified exact calculations, further close
`mu_4^C(8)=mu_4^Qbar(8)=mu_4^Q(8)=12`. These low-dimensional theorems do not
yet supply the missing 16D lower bound. Phase 27 is now **parked** at this exact
checkpoint. The 16D rank-22/23 branch remains an open mathematics problem and
has no current priority unless a later ADR identifies a direct FEMPS algorithm
or physics dependency.

ADR 0017 and Phase 28 restore algorithm and physics as the project main line.
The primary route is a restricted nonbranching diagonal-path FEMPS, stored as
`O(KDN)` orbital data and contracted through `K^2` determinant/Slater--Condon
transitions. This route is exactly antisymmetric and remains in first
quantization with continuous functional orbitals, but it is scientifically
close to nonorthogonal multideterminant/selected-CI methods. The E1--E4 ladder,
independent `D`/`K` convergence, three-seed stability, runtime/CPU-memory
measurements, and bounded CI/Slater/AGP/ordinary-TT study now pass for the
harmonic benchmark. At `D=6`, `K=2` reaches the finite-basis truth within
`2.21e-11`; all recorded symmetry residuals are zero. This is a restricted-
algorithm success and a measured tradeoff, not a novelty, asymptotic-
scalability, or superiority result. A generic stochastic route is only a
registered backup and must pass an explicit error/variance/failure-probability
gate before work begins.

Phase 32 extends that evidence to the interacting `N=6` soft-Coulomb model with
independent axes. At fixed `D=10`, `K=1,2,4` energies decrease from
`25.052242782725` to `25.049825287522`; at fixed `K=4`, `D=8,10,12` energies
decrease from `25.051264850894` to `25.049471144618`. The D12 error against
same-basis direct CI is `1.04729e-4`, with variance `1.12233e-3`. Raw exterior
coefficients independently reproduce every table without a `D^N` tensor, and
all admitted antisymmetry residuals are zero. Phase 33 subsequently removes the
Python transition/factor loops. At N6,D10,K4, CPU batched/reference differences
are `8.899e-16` in the Hamiltonian and `4.687e-13` in gradients; Blackwell/CPU
differences are `1.429e-14` and `3.682e-12`. The CPU forward/backward kernel is
`34.926x` faster than the pairwise reference. Both backends are admitted, but
CPU remains default because the matched full solve takes `5.0845 s` on CPU and
`11.9605 s` on Blackwell. These remain bounded numerical data, not an
asymptotic fit, generic-efficiency result, or GPU-acceleration claim.

Phase 35 repeats the truth-free K5/K6 selection with three fresh candidate-pool
seed pairs. All three lineages improve monotonically from the common D12,K4
source; their K6 energies span `4.87701e-6`, with maximum same-basis CI error
`3.76345e-5` and maximum variance `4.54299e-4`. Independent exterior
reconstruction verifies the six states, source nesting, selections, resources,
and zero structural antisymmetry residual without virtual-path or `D^N`
particle-tensor enumeration. All predicted and realized decisions say
`continue`. Because the preregistered experiment contains no stop event, it
does not validate automatic termination: production growth remains explicitly
bounded by an external maximum K. This is multiseed stability evidence at one
fixed physical point, not particle-number, basis, or asymptotic scaling.

Phase 36 moves this workflow out of benchmark scripts. A versioned public API
now requires a finite external `max_terms`, a complete consecutive candidate/
optimizer seed schedule, source and operator identities, and a checkpoint path.
It writes an atomic outer checkpoint after each K and rejects changed identities
on resume. Small-system tests verify materialized exterior energy, exact
antisymmetry, AD gradients, invalid schedules, and interrupted/uninterrupted
equivalence. The preregistered N6,D12 call stops after K5 and resumes K6,
reproducing Phase 35 lineage-1 selected candidates `12/3` and stored energies
exactly. Independent reconstruction gives final CI error `3.27575e-5` and
variance `3.89800e-4`. A first extra-QR wrapper attempt missed the frozen K6
energy by `3.60423e-11` and is retained as failed numerical evidence; the
`1e-11` gate was not changed. Phase 36 establishes reusable bounded solver
orchestration, not automatic stopping or new asymptotic evidence.

Phase 37 closes the historical-source gap. A versioned command constructs the
N4,D6 harmonic-oscillator basis, Q128 physical-SVD soft-Coulomb operator, and
canonical lowest-orbital Slater from explicit inputs. It optimizes K1 and then
runs the public externally capped K2--K4 schedule. A forced K2 interruption
and resume agrees exactly with a clean run at all four energies and candidates
`23/30/25`. Independent dense/exterior reconstruction gives final CI error
`4.88287e-10`, variance `2.86287e-9`, machine-precision norm, and zero
structural antisymmetry/path-enumeration residual. This is one bounded
end-to-end solver point; seed robustness, automatic stopping, and scaling
remain unadmitted.

Phase 38 closes the first clean-source seed-robustness gate at that same
N4,D6 point. Two preregistered schedules start from the canonical Slater and
select different K2/K3/K4 candidates (`1/13/22` and `31/26/25`). A forced K2
interruption/resume reproduces schedule A exactly. Independent exterior
reconstruction gives a Phase37/A/B final-energy spread of `2.035e-9`, maximum
fresh same-basis CI error `2.523e-9`, maximum variance `1.462e-8`, zero
optimizer failures, and zero antisymmetry or production-enumeration residuals.
This is two-schedule numerical evidence at one physical point, not universal
seed independence. ADR 0028 keeps it inside the single structural/no-go
manuscript as an NOCI-equivalent numerical exercise. A clean-source D axis may
support that audit but cannot open a separate method paper; the next scientific
gate is a non-NOCI explicit-correlation advantage or a matched Li--Waintal/
same-basis-DMRG tradeoff.

Phase 39 closes the associated publication-scope correction. The original
Theorems 1--3 are again displayed separately in the sole combined manuscript.
The Cayley construction proves hard pointwise evaluation at maximum
`chi=2`; the signed exact squared-norm reduction uses a direct-sum reference
and reaches maximum bond three. Exact squared-norm hardness promised at
`chi=2` therefore remains a conjecture. The diagonal-path data are retained
only as an NOCI-equivalent numerical exercise.

The bounded Phase 39 explicit-correlation prototype supplies **numerical
evidence** that a symmetric Gaussian correlator preserves antisymmetry and can
produce finite-basis projections whose Slater rank grows with projection
dimension. It is a `Q^2` materialization oracle at `N=2`, not a contraction
theorem, production solver, or `D`-convergence advantage. ADR 0029 selects this
as the sole candidate route, and ADR 0030 preregisters it against optimized
fixed-`K` NOCI, but that experiment is now parked. ADR 0031 first restores the
single clean-source `N=4,D=8` point from the original Phase 39 plan as internal
NOCI-equivalent evidence; no further small points are admitted. No second
manuscript exists: algorithm experiments must first produce and independently
reproduce a genuine differentiator.

ADR 0004 historically made ordered-sector/interparticle-distance functional TN
the primary numerical route; ADR 0017 supersedes that priority while retaining
it as a first-quantized control. Phase 15 and ADR 0005 replace its dense comparator by
a production native MPS/MPO contraction on the finite grid. The training path
does not materialize a local-dimension-to-the-number-of-sites tensor; dense
objects survive only as explicitly bounded post-training truth audits. This is
now extended by Gate G to one resource-closed controlled `N=8,D=12` point.
Incremental structured compression removes temporary dense raw Fourier-MPO
blocks, and the staged local action removes the pathological 78.12 GiB chi-32
intermediate. Bond 128 passes a gauge-fixed, many-body-normalized tangent audit
against bond 192; the original Gate F raw-coordinate miss remains historical
evidence rather than a changed threshold. The D12 production error against an
exterior D14 numerical reference is `7.174e-3`, a `17.6%` improvement over
Gate F D10 against the same reference. The constant-in-N Fourier channel count
still does not establish continuum or end-to-end asymptotic scaling, and the
descriptive N=2/4/6/8 trend does not admit N=10. The decision is numerical
evidence, not a method-priority claim.
