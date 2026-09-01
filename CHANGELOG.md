# Research changelog

## 2026-09-02 - Phase 46 external reproduction comparison closed locally

- Added a clean-rerun comparison tool that authenticates the primary Phase 44
  artifact, verifies an external result from its own six optimizer checkpoints,
  D6 clean control, and raw coordinate samples, and reconstructs every frozen
  gate decision.
- Replaced any cross-hardware bitwise-result expectation by the frozen
  uncertainty allowance `5 sqrt(SE_primary^2 + SE_reproduction^2) + 2e-4`,
  while still requiring the same selected lineages, aggregate failure,
  consecutive D4/D6 subgate, antisymmetry tolerance, and no forbidden
  materialization.
- A full self-comparison through the external-artifact path passes with zero
  energy difference and maximum antisymmetry residual `1.0401e-15`. It remains
  explicitly non-external: named-human identity, independence, conflicts, and
  checkpoint non-reuse cannot be certified by code.
- Phase 46 remains active awaiting actual human algebraic review and external
  numerical reproduction. No theorem status, manuscript split, or Paper B
  authorization changed.
- The standard repository suite passes `308` tests in 666.88 s, with the one
  known latticeTN report-path scalar-conversion warning.

## 2026-09-02 - Phase 45 declines an unmatched comparator table

- Reapplied ADR 0020 to Phase 44: at D4/D6/D8 the same-basis exterior spaces
  have dimensions 1/15/70 and are already exactly diagonalized, so
  second-quantized DMRG would only approximate a stronger existing CI truth.
- Audited the controlled Li--Waintal-style ordered-coordinate N4 point. Its
  COM/gap sine basis, box, interaction polynomial, MPS bond, 6,600 parameters,
  and `~4.4e-3` basis-dominated error are not matched to Phase 44's harmonic
  carrier D axis, five-feature correlator, or VMC uncertainty contract.
- Accepted ADR 0034 and declined both immediate calculations rather than
  creating a misleading table. No new FEMPS sample, NOCI point, larger size,
  DMRG run, or ordered-coordinate run was started.
- Closed Phase 45 and opened Phase 46 solely for human algebraic-complexity
  review and external Phase 44 reproduction handoff. Paper B remains closed.
- Added the Phase 44 clean external-run packet with frozen commits, commands,
  hashes, resource expectations, and mandatory failed-gate wording, plus a
  shared named-human sign-off template. No external sign-off is claimed.

## 2026-09-02 - Phase 44 interacting N4 gate retained as failed

- Ran exactly the ADR-0033 `N=4,D={4,6,8},P=5,chi=1` experiment from the
  disclosed Phase 37 K1 carrier. Six optimizer lineages and six blind
  selection evaluations completed before the comparator module was imported;
  selected lineages are `(2,1,2)`.
- The overall preregistered gate fails. Both D4 selection evaluations exceed
  the `2.5e-4` SE limit; all four D6/D8 selection evaluations have ESS
  `47,445--49,733`, slightly below the frozen 50,000 minimum. No rescue sample,
  threshold change, replacement seed, or additional point was run.
- All six held-out confirmations pass. Combined energies are
  `11.0243089336`, `11.0231801947`, and `11.0231265435`; conservative fixed-K4
  error ratios are `0.0320`, `0.3783`, and `0.8676`. The physical advantage
  subgate therefore passes at consecutive D4/D6 and fails at D8.
- Confirmed ESS is `2.67e5--2.89e5`, all R-hat values are within `3.1e-5` of
  one, maximum recorded antisymmetry residual is `1.041e-15`, and the forced
  D6 interruption/resume is bitwise identical to its clean trajectory.
- Added three hashed raw-coordinate archives, seven immutable optimizer
  checkpoints, a provenance manifest, and an independent verifier. It
  recomputes all 12 evaluations and every decision with zero current-
  environment observable difference.
- Closed Phase 44 without authorizing Paper B and opened Phase 45 only for a
  no-rescue matched-comparator feasibility decision.
- The standard repository suite passes `305` tests in 669.86 s with the one
  known latticeTN report-path scalar-conversion warning.

## 2026-09-02 - Preregistered interacting N4 explicit-correlation D gate

- Added a checkpointed CPU/float64 stochastic Adam/QR optimizer for the
  continuous explicit-correlator/exterior-Slater state. Minimal tests prove
  exact interrupted/resumed versus clean trajectories and reject changed
  checkpoint initialization identities.
- Accepted ADR 0033 before any interacting `N=4` coordinate-VMC production
  result. It freezes `D={4,6,8}`, `P=5`, two optimizer lineages, independent
  selection and held-out confirmation seeds, all sampler/optimizer budgets,
  a reference-use firewall, and a conservative uncertainty-aware advantage
  rule.
- Froze the disclosed Phase 37 `D=6,K=1,seed=3701` single-Slater carrier as
  the only preoptimization source. No multideterminant stage, CI vector, or
  reference energy initializes the state.
- Reused already-computed D4/D6/D8 CI and fixed-K4 NOCI comparators; no new
  ordinary NOCI point is authorized. Phase 44 must retain a failed outcome
  without changing axes, thresholds, states, or budgets.
- Added an exact reference-free initialization fixture containing only the
  Phase 37 K1 real carrier. Its verifier checks the full source hash and exact
  tensor equality; production is forbidden from opening the comparator-bearing
  Phase 37 artifact until lineage choices have been serialized and hashed.
- Closed Phase 43 at fixed-state estimator validation and opened Phase 44 for
  the one interacting differentiator experiment. Paper B remains closed.
- The standard repository suite passes `300` tests in 655.30 s with the one
  known latticeTN report-path scalar-conversion warning.

## 2026-09-02 - Phase 43 fixed-state coordinate-VMC validation

- Added a general-small-`N`, CPU/float64 coordinate-space VMC backend for a
  symmetric Gaussian correlator times a continuous exterior Slater carrier.
  It evaluates analytic local energies and covariance gradients without a
  `D^N` coefficient tensor, full alternating tensor, or virtual-path scan.
- Implemented deterministic multi-chain sampling, autocorrelation/ESS,
  blocking and chain uncertainties, R-hat, sampled antisymmetry residuals,
  atomic checkpoints, and exact RNG-aware resume.
- Passed the frozen ADR-0032 validation. Two `N=2` runs agree with `Q=160`
  deterministic truth within preregistered uncertainty, with ESS above
  `8.3e4`, R-hat below `1.000016`, and antisymmetry residual below `3.84e-16`.
  The `N=4` noninteracting run gives energy exactly 8, variance `4.96e-31`,
  and bitwise-identical interrupted/resumed and clean samples.
- Added a hashed 2.9 MB raw-coordinate archive and independent verifier that
  recomputes every stored observable and gradient with zero current-
  environment difference and independently repeats the `N=4` resume path.
- Retained the scientific boundary: the near-stationary `N=2` gradient test is
  an absolute-error implementation check; no interacting `N=4` result,
  external replication, scalable-solver claim, or Paper B is admitted.
- The standard repository suite passes `298` tests in 655.68 s with the one
  known latticeTN report-path scalar-conversion warning.

## 2026-09-02 - Phase 40 explicit-correlation differentiator and clean reproduction

- Resumed only the preregistered non-NOCI differentiator after manuscript-A's
  internal theory closure; no additional standalone NOCI point or second-paper
  source was created.
- Added a deterministic `N=2` optimizer for a symmetric Gaussian correlator
  times a `chi=1` continuous exterior carrier. Carrier orbitals and correlator
  amplitudes are both differentiated, QR gauged, checkpointed, and audited by
  materialization and central finite differences.
- Ran the frozen Cartesian axes: 72 correlated `D/P/seed` points and 54
  fixed-`K` NOCI controls. All antisymmetry, correlator symmetry, norm,
  quadrature, gradient, exact finite-basis contraction, and no-virtual-path
  checks pass.
- Under the frozen `P=5` versus `K=4` rule, `D=2,4,6,8` pass and `D=10,12`
  fail. The consecutive passing pairs are `(2,4)`, `(4,6)`, `(6,8)`; the
  high-`D` failures and seed spread are retained rather than rescued.
- Independently reconstructed all 126 serialized states. Correlated
  observables reproduce exactly, NOCI observables within `4.441e-15`, and the
  maximum AD/finite-difference discrepancy is `7.585e-11`.
- Repeated every optimization from a new clean checkpoint tree. All compared
  energies, variances, norms, residuals, uncertainties, and gate decisions
  agree exactly with the primary run. This is repository-level deterministic
  reproduction, not external scientific replication or a scalable
  many-particle contraction result; Paper B remains closed.
- The full repository suite passes `292` tests in 654.47 s with the one known
  latticeTN report-path scalar-conversion warning.

## 2026-09-02 - Manuscript-A theory recovery and CHSS bond-two correction

- Kept the restored preregistered `N=4,D=8` result as internal numerical
  evidence only. It failed the registered final CI-error and variance
  thresholds, so no rescue or additional small NOCI-equivalent point was run.
- Audited CHSS Theorems 3.5 and 3.9 against the primary paper. Their structured
  output `a I_2+b J_2`, `a+b=4^(3m)#SAT`, sharpens exact rational FEMPS
  squared-norm hardness from maximum bond three to maximum bond two by the
  fixed boundary `u=e_1`, `v=e_1+e_2`. Bond-three polarization is retained only
  for arbitrary signed Cayley outputs.
- Restored a self-contained exact-TT rank proof, expanded the matrix-pair
  collapse proofs with rational construction and bit-complexity conditions,
  and made the direct-product carrier counterexample and scalar/noncommutative
  determinant distinction explicit in the single combined manuscript.
- Replaced the submission-facing restricted-solver survey by one selected
  `N=6,D=12,K=4` result in the 924-dimensional exterior space, explicitly
  disclosing its zero-padded preoptimized `D=10,K=4` initialization and its
  NOCI-equivalent status.
- Added an internally complete proof draft for exact unnormalized pointwise
  `#P`-hardness in the standard unnormalized rational Legendre basis, including
  an inverse-evaluation-matrix bit bound and one-query metric reduction. The
  submission claim remains a conjecture pending external human
  algebraic-complexity review.
- Added a primary-source CHSS audit, an independent AI audit clearly labeled
  as non-human, an external-review packet, a small exact Legendre verifier,
  and a primary NOCI reference.
- Built and visually inspected the 15-page combined v5 PDF with no undefined
  references, LaTeX warnings, clipping, overlap, or placeholder figures. The
  final standard suite passes `287` tests with one known latticeTN report-path
  warning; the exact-certificate subset passes `10` tests.

## 2026-09-02 - Restored Phase 39 N4,D8 preregistration

- Restored the original clean-source Phase 39 obligation as exactly one
  `N=4,D=8,Q=128,K1--K4` internal NOCI-equivalent calculation, without
  deleting the later manuscript-scope audit that reused the phase number.
- Accepted ADR 0031 and froze source seed `4001`, candidate/optimizer pairs
  `4011/4012`, `4021/4022`, `4031/4032`, inherited D6 optimizer budgets,
  accuracy/resource gates, forced K2 interruption/resume, and clean repeat
  before any D8 production result.
- Parked the Phase 40 explicit-correlation experiment. After the single D8
  artifact is independently verified, no additional small numerical point is
  admitted and the main line returns to manuscript-A theory closure.
- Ran the sole registered D8 schedule. Clean/resume energies and candidates
  agree exactly and all structural/resource gates pass, but the final CI error
  `3.6469e-5` and variance `3.2252e-4` fail their `1e-6`/`1e-5` gates. The
  independently reconstructed failure is preserved without a rescue run.

## 2026-09-02 - Phase 39 closure and no-paper-before-result rule

- Closed Phase 39 with one authoritative combined manuscript. The former
  restricted-method draft remains only a frozen reproduction note; no second
  manuscript is in development.
- Confirmed that Structural results I--III and the exact `chi=2`/maximum-bond-
  three distinction are directly visible in the combined source. The apparent
  disappearance came from an earlier consolidation/relabeling, not from a
  retraction of the results.
- Added an independently reproducible exploratory `N=2` symmetric-correlator
  exterior-carrier prototype. It has zero audited antisymmetry residual and
  passes AD/materialization checks, but does not establish an advantage over
  optimized NOCI and therefore makes no method-paper claim.
- Opened Phase 40 solely as a preregistered algorithm experiment with separate
  `D`, `P`, and fixed-`K` axes. ADR 0030 forbids second-manuscript drafting
  until an independently reproduced non-NOCI differentiator passes.
- Passed the standard suite with `283 passed, 1 known latticeTN report-path
  warning` in 680.47 s. The CPU 2201 baseline reproduced
  `E=1.8788029184435575`, with absolute reference error `1.08054e-4`.

## 2026-09-02 - Single-manuscript restoration and distinctiveness gate

- Accepted ADR 0028 and withdrew the two-paper split. The structural/no-go
  manuscript is again the sole submission candidate; current diagonal-path
  results are retained only as an NOCI-equivalent numerical section.
- Restored the original three-result theory chain as explicit Structural
  results I--III and made the fixed-bond boundary visible: the hard pointwise
  amplitude uses `chi=2`, the proved signed exact-norm reduction reaches maximum
  bond three, and the `chi=2` exact-norm sharpening is a conjecture.
- Froze the standalone restricted-method draft as an internal working note.
  Any future method paper now requires either a non-NOCI explicit-correlation
  functional-basis `D`-convergence advantage or a matched Li--Waintal/
  same-basis-DMRG comparison with a genuine measured tradeoff.
- Replaced the planned N4,D8-first Phase 39 priority by combined-manuscript
  closure and a bounded audit of at most two differentiating algorithm routes.
- Rebuilt the 14-page combined PDF, visually inspected every page, and passed
  the standard suite: `277 passed, 1 known latticeTN report-path warning` in
  688.51 s.

## 2026-09-02 - Phase 38 clean-source seed robustness closure

- Preregistered two fresh complete N4,D6 clean-source schedules before
  production, retaining the Phase 37 model, K1--K4 budgets, tolerances, public
  command contract, and external K cap.
- Forced schedule A through a K2 interruption/resume and clean repeat with
  exact energy/candidate agreement; schedule B completed cleanly with a
  distinct candidate path. No outcome-dependent retry occurred.
- Independently reconstructed all 12 complete-run exterior states. Including
  Phase 37, the final-energy spread is `2.035e-9`; maximum fresh CI error is
  `2.523e-9`, maximum variance `1.462e-8`, and optimizer failures, structural
  antisymmetry residuals, and production enumeration counts are zero.
- Expanded the numerical reproduction manifest to 15 independently verified
  artifacts and closed Phase 38. ADR 0028 subsequently replaced the planned
  N4 D6-to-D8-first priority with combined-manuscript closure and a
  distinctiveness audit.

## 2026-09-01 - Phase 37 clean Slater-source solver closure

- Added a public command that constructs the registered N4,D6,Q128
  soft-Coulomb model and canonical lowest-orbital Slater from explicit inputs,
  optimizes K1, and enters the bounded adaptive K2--K4 solver without a
  historical FEMPS checkpoint or CI initialization.
- Added versioned command/result checkpoints with full configuration,
  source/operator identity, atomic stage writes, and strict resume validation.
  A forced K2 interruption followed by resume agrees exactly with a clean run
  at every K.
- Independently reconstructed the final exterior state at
  `E=11.023837713691632`, same-basis CI error `4.883e-10`, variance
  `2.863e-9`, norm error `3.33e-16`, and zero structural antisymmetry residual,
  with no virtual-path or production `D^N` materialization.
- Expanded the restricted-method reproduction manifest to 14 independently
  verified numerical artifacts, closed Phase 37, and opened Phase 38 for two
  preregistered clean-source seed schedules at the same physical point before
  any N/D expansion.

## 2026-09-01 - Phase 36 public adaptive solver closure

- Promoted the bounded adaptive diagonal-path solver to a public API with an
  externally required maximum `K`, complete candidate/optimizer seed schedules,
  versioned atomic stage checkpoints, interruption/resume, and strict
  source/operator identity checks.
- Reproduced the frozen first `N=6,D=12` candidate-pool lineage exactly across
  a forced `K=5` interruption. The accepted serialized `K=5` and `K=6`
  energies match Phase 35 exactly; the preserved first attempt records a
  `3.60e-11` mismatch caused by duplicate source gauge projection.
- Independently reconstructed the final exterior state at
  `E=25.049399173588696`, CI error `3.276e-5`, variance `3.898e-4`, norm error
  `1.11e-16`, and zero structural antisymmetry residual, with no virtual-path
  or production `D^N` materialization.
- Expanded the restricted-method reproduction manifest to 13 independently
  verified numerical artifacts, closed Phase 36, and opened Phase 37 for a
  clean end-to-end single-Slater-source command without historical checkpoints.

## 2026-09-01 - Phase 28 interacting algorithm gate passed

- Closed E4 for the restricted diagonal-path FEMPS with three blind `D=6,K=4`
  seeds and three truth-free nested-basis `D=7,K=4` continuations; all six pass
  registered energy, variance, norm, antisymmetry, and no-enumeration criteria.
- Added independent `K=1,2,4` and `D=5,6,7` convergence, sampled whole-process
  CPU RSS, and exact-CI, Slater, single-AGP, and ordinary particle-TT
  comparators. The standalone verifier recomputes the raw acceptance decision.
- Added a condition-gated determinant/inverse transition path with automatic
  singular-safe minor fallback. Ten `(N,D,K,L)` points agree within `1.421e-14`;
  the tested `N=4` forward/backward kernels improve by roughly `2--4x`.
- Recorded the narrow decision: the subclass is a successful exact restricted
  baseline with systematic correlation control, but the tiny benchmarks show
  no CI runtime/parameter advantage and admit no generic, novelty, asymptotic,
  or superiority claim.

## 2026-09-01 - FEMPS algorithm and physics recovery pivot

- Parked Phase 27 after the exact seven- and eight-dimensional four-form
  checkpoints; retained the 16D rank-22/23 branch as an open problem rather
  than an active algorithm milestone.
- Revised the master plan so generic polynomial exact contraction is no longer
  the default success condition and high-dimensional rank classification no
  longer receives independent main-line priority.
- Opened Phase 28 and accepted ADR 0017. The primary route is an exactly
  contractible nonbranching diagonal-path FEMPS (`K` nonorthogonal Slater
  paths); the only registered backup is a separately gated controlled
  stochastic estimator for generic cores.
- Added the algorithm-feasibility audit, explicit complexity contract, ordered
  E1--E4 physics gates, mandatory antisymmetry/error/runtime reporting, and the
  third-draft paper track.
- Implemented singular-overlap-safe `K^2` determinant contractions for
  diagonal-path FEMPS norms, one-body and factorized two-body transitions,
  plus variable-projection orbital optimization and checkpoint/resume.
- Added independent exterior value/gradient tests and the first reproducible
  E1/E2 ladder. At `N=2,D=6,kappa=0.35`, increasing `K=1,2,4` reduces the
  continuum-energy error from `2.824e-4` to `5.927e-5` and `1.306e-5`, with
  zero measured antisymmetry residual; these are numerical evidence only.
- Passed the E3 representation check at `N=4,D=6,K=1`: exact energy and zero
  variance/residual coexist with ordinary particle-TT ranks `(4,6,4)`.
- Added an E4 harmonic-interaction pilot with independent `D=5,6,7` and
  `K=1,2,4` trends plus exact same-basis references. E4 remains open pending
  stronger high-`D` optimization, CPU peak-memory instrumentation, and the
  required comparator table.
- Repeated the representative E4 `D=6,K=4` point on the RTX PRO 4000
  Blackwell: CPU/GPU energies agree within `7.75e-13`, peak allocated GPU
  memory is `30,736,384` bytes, and the small determinant loops show no GPU
  speed advantage.

## 2026-09-01 - Phase 27 four-form reconstruction checkpoint

- Reconstructed the working extremal question as the minimum middle
  contraction rank of a concise four-form, while labeling both the notation
  `mu_4(m)` and the 16D 22/23 alternatives as conjectural until their missing
  provenance is recovered.
- Audited primary contraction-rank, essential-variable/radical, and skew-rank
  sources, and separated middle catalecticant rank from Grassmann/Slater
  decomposition rank.
- Added standard-library-only exact rational and recorded-prime contraction
  utilities with tests for symmetry, complementary cuts, basis permutations,
  Hodge duality, support loss in dimension five, the six-dimensional
  symplectic-dual control, and direct sums.
- Certified the explicit rational 16D direct sum with rank vector
  `(1,16,24,16,1)` using an implementation-independent verifier and payload
  hash `3e48d8e9e0ed1802805d5446c573cef7daca05146abae45d90679fa5a633edcd`.
- Screened 120,000 seeded coordinate-hypergraph samples over `F_2`; 118,317
  were concise and none had middle rank below 24. This is recorded only as
  numerical evidence and supplies no lower bound or orbit/chart coverage.
- Proved the exterior-apolar identity
  `Hilb(Lambda(V)/Ann_wedge(omega)) = (rank C_j(omega))`, separating the
  canonical four-form quotient from arbitrary formal-dimension-four PD
  algebras.
- Closed the first nontrivial low-dimensional extremum:
  `mu_4^Q(7)=mu_4^Qbar(7)=12`. Cohen--Helminck supplies exhaustive nine-orbit
  coverage, while a standard-library-only independent verifier checks the
  transcribed representatives, all exact contraction ranks, and payload hash
  `94f1a654978dd1d37770b5a2171a07a5a839525dac1d16b6247a3b1ab2665f21`.
- Closed the eight-dimensional extremum:
  `mu_4^C(8)=mu_4^Qbar(8)=mu_4^Q(8)=12`. Antonyan--Oeding's Cartan subspace and
  94 nilpotent normal forms, joined by the theta-group orbit-closure theorem,
  reduce the lower bound to exact finite checks. The independent verifier
  recomputes all 94 rank vectors, 28 Cartan joint eigenpairs, the `F_3`
  hyperplane bound, and payload hash
  `44288f6097c7f56c746f3e3c39885fe707704acf47b957129e786afab044214b`.
- Ruled out ordinary commutative socle-degree-four Gorenstein Hilbert vectors
  as the missing source of the 16D alternatives: their classified `r=16`
  minimum is 15 and belongs to a different symmetric-algebra problem.

## 2026-09-01 - Phase 26 manuscript v1 and fixed-bond proof audit

- Replaced the provisional fixed-small-bond conjecture by a direct theorem:
  site-indexed one-form cores encode the row-ordered Cayley determinant at
  bond two, and a scalar-reference direct sum recovers its signed amplitude
  from two exact squared-norm queries at maximum bond three.
- Added an implementation-independent exact-integer verifier for orders 2--6,
  all four matrix boundaries, the bond-three polarization construction, and
  exact antisymmetry, with certificate hash
  `1d2208d3e5cb14f5c8e6c875f7fddf51c47ce9a3e61be6cedb8246d662b3a016`.
- Added the production one-form Cayley-core constructor and materialization/
  squared-norm regressions for orders 2--4.
- Assembled R1--R2 and C1--C6 into an 11-page internal synthesis manuscript,
  including a proof-audit ledger, full evidence hashes, limitations, and an
  expanded AGP/number-projection/Pfaffian/mVMC/fermionic-entropy bibliography.
- Compiled the manuscript with resolved citations and no LaTeX, box, or layout
  warnings; a full-page contact-sheet inspection found no clipping or broken
  page flow.
- Passed the complete `209`-test suite with one unchanged latticeTN report-path
  warning and issued ADR 0016 for external proof review.

## 2026-09-01 - First-draft reviewer-response handoff

- Prepared a bilingual reviewer-response memorandum to accompany the second
  draft, separating accepted, qualified, and not-adopted recommendations.
- Synchronized the response with the Phase 24 approximation boundary and the
  closed Phase 25 statistics-carrier result.
- Recorded fixed-small-bond noncommutative-determinant hardness as a conjecture
  at handoff; the subsequent Phase 26 audit resolved it by the direct
  site-indexed one-form construction above.
- Passed document structural and accessibility audits with zero reported
  issues; page-render inspection remains unavailable in the current
  environment because LibreOffice is not installed.

## 2026-09-01 - Phase 25 statistics-carrier/multiplicity Gate L

- Defined the proposed cut object as the image of the exterior contraction map
  rather than an informal “reduced bond.”
- Proved that Slater multiplicity one forces a one-cut carrier of dimension
  `N`, while a two-Slater full-support family has rank `N+2`; hence no
  universal free tensor product exists for `N>=3`.
- Separated valid symmetry-adapted structural/degeneracy tensors from the
  failed analogy: particle antisymmetry has a one-dimensional sign irrep and
  `GL(V)` retains the full exterior irrep.
- Classified state-adaptive occupied spaces and Slater/secant decompositions as
  established, nonfree determinant-channel structures rather than a canonical
  FEMPS multiplicity spectrum.
- Reapplied the sparse-path permanent example: its output is projectively one
  Slater, yet compact-input scale/norm contraction remains hard.
- Added exact all-cut, component-locking, orbital-permutation, direct-embedding,
  and perturbation controls for `3<=N<=8`, with certificate hash
  `06025f168a49c0ab857c2163103ffabcb56fb04cd1fed4df9120d25ef6bc60df`.
- Compiled a four-page theorem draft without layout/reference warnings and
  passed the complete `205`-test suite with one unchanged latticeTN warning.
- Closed Gate L and advanced to a unified manuscript v1/proof-audit phase.

## 2026-09-01 - Phase 24 controlled approximate exterior Gate K

- Separated exact permanent hardness from approximation complexity; the 0--1
  Phase 22 reduction is compatible with the established FPRAS for entrywise-
  nonnegative matrices.
- Proved a direct real-PSD transfer: a relative sparse-path squared-norm PRAS
  would yield a PRAS for real-PSD permanents and therefore imply `RP=NP`.
- Classified Gurvits-type additive estimates as conditionally useful only when
  a positive norm lower bound keeps the Rayleigh denominator certified.
- Derived a simultaneous norm/numerator energy-error bound and endpoint
  confidence interval, explicitly separating estimator unbiasedness from ratio
  bias and tail control.
- Added exact positive, cancelling, precision-ill-conditioned, signed-PSD, and
  energy-bound controls with certificate hash
  `c15e7ff268a962e2790004c7f63d47bedb53be0c887ab0241e671b7fe4ff3b16`.
- Audited APG selection, PSD/nonnegative permanent approximation, stochastic
  TN contraction, and sign-problem literature; no prior-art component is
  claimed as new.
- Compiled the expanded six-page theorem draft without layout/reference
  warnings and passed the complete `204`-test suite with one unchanged
  latticeTN report-path warning.
- Closed Gate K and advanced to a canonical statistics-carrier/correlation-
  multiplicity Gate L; no post-failure GPU/AD solver was admitted.

## 2026-09-01 - Phase 23 exact exterior no-go synthesis

- Separated ordinary particle-TT exchange rank from compact exterior
  contraction complexity in one theorem dependency and coverage map.
- Promoted the bandwidth-one APG permanent identity to the simplest generic
  exact squared-norm obstruction while retaining the tagged Cayley theorem as
  an independent growing-order-memory result.
- Corrected stale Gate A and Phase 14 statements, especially the former
  conflation of fixed `Mat_2` pair powers with row-ordered determinants.
- Added manuscript-safe/forbidden claim language, a paper outline, full
  certificate hashes, and explicit field/normalization/reduction qualifiers.
- Added a production matrix-pair and one-form FEMPS regression for the sparse
  permanent identity.
- Compiled both theorem drafts successfully; the updated contraction draft has
  resolved references and no layout warnings.
- Passed the complete `203`-test suite with one unchanged latticeTN report-path
  warning and advanced to a controlled-approximation Gate K.

## 2026-09-01 - Phase 22 sparse growing-width path gate

- Identified the upper-bidiagonal endpoint pair-matrix state exactly as an
  antisymmetrized product of geminals rather than a new exterior ansatz.
- Corrected the proposed formal-monomial Waring argument: ordinary Waring rank
  does not descend automatically through exterior multiplication to a physical
  LC-AGP lower bound.
- Proved that paired-orbital edge forms encode an arbitrary matrix permanent in
  the unique-path top coefficient and its square in the normalized exact norm.
- Added an independent exact certificate covering three matrix families for
  every `1<=M<=6`; path, exterior-subset, and permutation routes agree with hash
  `dd72c1aaeb0bc2a6b9206992cde9099f2f568b7ff6c8ed8eb7e38d958f78e790`.
- Audited APG/APIG, Fischer AGP expansions, geminal RDM contraction, and
  permanent prior art; no priority claim is made for these structures.
- Closed Gate J negatively for generic tridiagonal/fixed-bandwidth growing
  virtual width and advanced to a consolidated exterior no-go theorem package.

## 2026-09-01 - Phase 21 one-generator growing-radical checkpoint

- Proved that arbitrary boundaries of `C[z]/(z^d)` matrix-pair powers collapse
  exactly to at most `M(d-1)+1` scalar AGPs, jointly polynomial even for growing
  radical depth.
- Added an implementation-independent exact-rational certificate covering all
  boundary basis functionals for every `1<=M,d<=4`.
- Separated exact Waring rank from border/cactus rank and audited the result
  against monomial Waring, Veronese osculation, curvilinear schemes, moment
  tensors, and border-rank-motivated AGP-CI.
- Rejected the one-generator commutative path algebra without numerical solver
  development; the next candidate must use controlled multibranch
  noncommutative growing memory.
- Embedded the minimal noncommutative alternating-word algebra of dimension
  `2d-1` into `Mat_2(C[z]/z^d)` and derived the exact bound
  `[M(d-1)+1] binom(M+3,3)`.
- Added a direct-word versus nested-interpolation exact certificate for every
  boundary word at `1<=M<=3,1<=d<=4`.
- Generalized the collapse to fixed matrix-state width over a fixed number of
  commuting grading counters and closed Gate I negatively for that class.

## 2026-09-01 - Phase 20 bounded coefficient-algebra classification

- Re-audited the Phase 13 tagged Cayley obstruction against the original
  noncommutative-determinant theorem and corrected a stale novelty-matrix entry.
- Proved that arbitrary-boundary `2 x 2` upper-triangular matrix-pair powers
  collapse to at most `binom(M+2,2)+2` scalar AGPs, even for genuinely
  noncommuting coefficients.
- Added an exact rational interpolation construction, an implementation-
  independent M=1--6 certificate, complex128 exterior equivalence, and
  restricted reverse-mode gradient tests.
- Proved in theorem-draft form that fixed semisimple block size and fixed
  radical nilpotency index imply a polynomial-size exact LC-AGP expansion,
  including noncommutative semisimple quotients.
- Added a complementary exact-rational M=1--4 `Mat2` certificate with term
  bound `binom(M+3,3)` and closed Gate H negatively for the entire uniformly
  bounded coefficient-algebra candidate class.

## 2026-09-01 - Phase 19 resource-safe N=8 Gate G

- Reordered latticeTN's two-site effective-Hamiltonian action into four bounded
  contractions, removing a 78.12 GiB path; formal chi-32 N=8,D=10 DMRG peaks at
  736,884,736 CUDA bytes and converges across two sweeps to `3.68e-11`.
- Added left-gauge, many-body-normalized MPS physical tangent directions and
  finite-difference tests. MPO bond 128 passes all tangent and energy budgets
  against bond 192 while the Gate F raw-coordinate miss remains recorded.
- Matched seed/schedule training at MPO bonds 128, 160, and 192; all pass the
  reference-energy and 2 GiB budgets, retaining 128 as the smallest production
  bond.
- Extended the independent N=8 exterior reference to D=14; Q128/Q160 differs by
  `9.24e-13`, with the value still labeled numerical rather than a continuum
  bound.
- Completed a blind N=8,D=12 multiscale run at `(ell,rho)=(0.55,3.0)`. Its
  `7.174e-3` error against D14 is `17.6%` below Gate F D10 against the same
  reference, and all Fourier/quadrature/optimizer/memory controls pass.
- Reassessed N=2/4/6/8 accuracy and resources. Gate G passes at controlled N=8
  scope, but N=10 and favorable asymptotic scaling remain unadmitted.
- Issued ADR 0009 and advanced the active plan to a restricted exterior-
  correlation gate beyond known finite LC-AGP/Gaussian structure.

## 2026-09-01 - Phase 16 continuous ordered-distance Gate D

- Derived the exact unit-Jacobian center-of-mass/positive-gap map, its Cartan
  kinetic metric, harmonic metric, `sqrt(N!)` chamber normalization, and exact
  signed fermion recovery.
- Added finite sine-box and unbounded odd-Hermite collision-Dirichlet bases with
  analytic/projected derivative, kinetic, position, and position-square
  operators plus independent quadrature tests.
- Built native continuum noninteracting and soft-Coulomb MPS/MPO operators. The
  direct interaction automaton has conservative raw bond `O(N^2 K)` and never
  powers a truncated position matrix.
- Added arbitrary-local-dimension latticeTN MPS initialization and blind global
  AD training without a product-basis gather.
- Closed basis/box, interaction-degree, quadrature, MPS-bond, optimization, and
  CPU/GPU parity axes with independent dense or matrix-free truth audits.
- Three blind N=2 runs reach their Galerkin truth within `3.01e-6`; three blind
  N=4 Blackwell runs lie `4.37e-3--4.39e-3` above an exterior `D=14` numerical
  reference and only `2.84e-5--4.63e-5` above same-basis Galerkin truth.
- Re-read Hong et al. and Li--Waintal at implementation level. The result is
  classified as a controlled continuous functional/operator/AD integration,
  not a priority claim for ordered coordinates or distance MPS and not FEMPS.
- Issued ADR 0006: Gate D passes at controlled `N<=4` scope. Phase 17 targets an
  unbounded interaction basis, global MPO compression audits, and N=6 scaling.

## 2026-09-01 - Phase 12 controlled soft-Coulomb benchmark matrix

- Added a normalized benchmark schema with exact closure of correlation,
  operator-representation, basis, and total error axes.
- Extended direct soft-Coulomb truth and finite-AGP evidence through N=8,
  D=12, and K=6, with reproducible K-growth probes and Blackwell resource data.
- Added a generic checkpoint-continuation runner and arbitrary supplied
  functional-operator support in the finite-AGP optimizer.
- Matched finite AGP, ordered-sector, exterior-CI, and ordinary particle-TT
  diagnostics on one identical finite-difference soft-Coulomb Hamiltonian.
- Found prior deterministic LC-AGP/AGP-CI work that overlaps the finite-AGP
  state family and K-squared contraction organization; reclassified finite AGP
  as a fallback/control rather than sufficient FEMPS novelty.
- Advanced the active work to a beyond-LC-AGP matrix-wedge structure gate.

## 2026-09-01 - Phase 11 gauge-balanced finite-AGP conditioning

- Reinterpreted the Phase 10 D10 raw overlap condition `143.5` as a scale-gauge
  artifact: the unit-diagonal overlap condition is only `1.464`.
- Added gauge-balanced whitening/compression, raw and balanced diagnostics, an
  invariant contribution Gram spectrum, and auditable leave-one-out pruning.
- Verified whitening and exact duplicate pruning against explicit exterior
  states; production runs discarded or restarted no directions.
- Refined three independent D10,K4 chains to errors `1.334e-5`, `1.900e-5`,
  and `2.031e-5`, all matching or improving the Phase 10 result.
- Reproduced K5 improvement on all three chains, reaching `6.000e-6`,
  `8.820e-6`, and `8.026e-6` with balanced conditions below `3.11`.
- Advanced the active work to a controlled N/D/K soft-Coulomb benchmark matrix.

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
- Started Phase 8 with a non-materialized `N=8,D=10` Blackwell benchmark. Blind
  polynomial K=1 training reaches energy `32` within `2.81e-13` of an
  independent 45-dimensional exterior truth while avoiding the `10^8`-entry
  ordinary particle tensor.
- Defined the normalized ordered-coordinate isometry, collision-wall boundary,
  distance-coordinate kinetic operator, and a three-way comparison protocol.
  A local harmonic-grid oracle agrees entrywise with independent exterior
  Hamiltonians for four `N=3` grid sizes.
- Completed the interacting `N=8,D=10,kappa=0.02` point without ordinary-tensor
  materialization. Single-AGP continuation reaches `3.709e-6` finite-basis
  error with polynomial/exterior agreement at `7.82e-14`.
- Gate B keeps finite-AGP FEMPS as the E6 production solver and the exact
  ordered-sector grid path as a complementary small-system control; no
  asymptotic superiority claim is made.
- Started E6 with a factorized Gauss--Hermite soft-Coulomb operator. Direct
  `Q=96` and `Q=128` tensors agree to `2.25e-11`; a `1e-14` kernel threshold
  retains 50 of 128 modes at `D=8` with `6.54e-13` dense reconstruction error.
- The `N=2,D=16` exterior energy `2.553831818868` agrees within `8.49e-8` with
  an independent second-order-extrapolated odd relative-coordinate grid result.
  Polynomial AGP energies and gradients pass explicit exterior checks.
- Blind `N=2,D=12` soft-Coulomb training resumes from step 200 and reaches the
  finite-basis truth within `2.13e-14`, with fidelity one and independent
  exterior agreement at `2.09e-14`. The resumed 600 steps take 490.9 seconds,
  exposing factor-axis batching as the next performance requirement.
- Replaced per-factor nested autograd by a batched mixed-derivative Newton
  recurrence. A matched N=2 per-step timing improves by 19.9x to `0.0411`
  seconds per step without changing explicit energy or gradient tests.
- Blind/restarted `N=4,D=8,Q=96,K=1` training reaches `1.446e-3` finite-basis
  error and `0.9996928` fidelity, with polynomial/exterior agreement at
  `1.60e-14`; finite-AGP growth remains necessary.
- Soft-Coulomb N=4 greedy K=2 reaches `7.445e-5` finite-basis error with stable
  overlap conditioning; a safe `N=6,D=8,K=1` attempt reaches `1.725e-3` error
  and polynomial/exterior agreement at `1.19e-12` in 33.7 seconds.
- Completed the N=4 soft-Coulomb D/K hierarchy. D=8 errors fall through
  `1.446e-3, 7.445e-5, 1.956e-5, 5.407e-6` for K=1--4; three independent
  shorter K=4 chains reproduce `9.19e-6`--`1.27e-5` without overlap-rank loss.
- Direct exterior truth reaches D=14. Best total differences from that numerical
  reference fall from `2.015e-4` at D=8,K=4 to `7.131e-5` at D=10,K=4.
- N=6 greedy K=2 reaches `1.484e-6` finite-basis error. Matched current-kernel
  timing shows only 2--4% soft-Coulomb time overhead over the two-factor harmonic
  interaction, though peak memory grows by 44%.
- Expanded the novelty audit to electronic Pfaffian QMC, FermiNet, PauliNet, and
  hidden-fermion Pfaffian states; broad first-quantized/Pfaffian priority claims
  are explicitly rejected.
