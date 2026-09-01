# Theory and evidence status

| Claim | Status | Required next evidence |
|---|---|---|
| A decomposable normalized Slater state has a flat particle-cut Schmidt spectrum | proved in draft and numerically verified | External proof review and literature cross-check |
| Ordinary particle TT must pay the exchange-statistics Schmidt multiplicity | proved in draft and numerically verified | Stronger concise/full-support bounds remain open |
| Matrix-wedge FEMPS is associative and strictly antisymmetric | proved from exterior associativity; small-N verified | External algebra review |
| Generic nontrivial FEMPS has polynomial exact contraction | conditionally obstructed by two exact reductions: Phase 13 embeds permanent-hard `Mat_2` Cayley determinants using growing order tags, while Phase 22 directly embeds a 0--1 permanent in the squared norm of a bandwidth-one, unique-path APG state; both map with polynomial bond into one-form FEMPS | External proof review; only separately gated restrictions or controlled approximation remain open |
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

Phase 13 strengthens the generic status from “no recurrence known” to a
conditional algebraic-complexity obstruction. The unrestricted dense
matrix-wedge ansatz cannot be the promised exact polynomial solver unless the
permanent admits a polynomial algorithm. This does not apply automatically to
additional structured core algebras or approximate contraction.

ADR 0004 makes ordered-sector/interparticle-distance functional TN the primary
numerical research route. Phase 15 and ADR 0005 replace its dense comparator by
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
