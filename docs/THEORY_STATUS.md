# Theory and evidence status

| Claim | Status | Required next evidence |
|---|---|---|
| A decomposable normalized Slater state has a flat particle-cut Schmidt spectrum | proved in draft and numerically verified | External proof review and literature cross-check |
| Ordinary particle TT must pay the exchange-statistics Schmidt multiplicity | proved in draft and numerically verified | Stronger concise/full-support bounds remain open |
| Matrix-wedge FEMPS is associative and strictly antisymmetric | proved from exterior associativity; small-N verified | External algebra review |
| Generic nontrivial FEMPS has polynomial exact contraction | conditionally obstructed: a polynomial-size tagged reduction embeds permanent-hard `Mat_2` Cayley determinants into a top-degree matrix-pair state and then into one-form FEMPS | External proof review; restricted algebras remain open |
| Fixed-number Pfaffian/AGP FEMPS contracts polynomially | theorem/algorithm draft; value, gradient, scaling and GPU verified; the LC-AGP family and K-squared organization have direct prior art | Use as fallback/control, not central novelty |
| Finite-AGP overlap compression can be made term-gauge invariant | proved by unit-diagonal balancing; explicit exterior and three-seed D10 tests pass | Relation of the contribution Gram spectrum to physical correlation remains open |
| 2201 functional MPS baseline transfers to the current backend | validated by controlled `D`/`chi`/seed scan | Larger paper-figure digitization is optional |
| Ordered-distance MPS/MPO has polynomial native contraction on the finite grid | Gate C pass: exact gap bijection, hard charge, kinetic/trap/soft-Coulomb MPO with raw bond `O(N^2(L-N))`; three blind N4,L8 runs reach `1.30e-6`--`2.07e-5` energy error | Continuous bridge is addressed by Gate D; finite-grid scaling remains an independent control |
| Continuous ordered-distance functional MPS/MPO is a controlled 2201 bridge | Gate F pass at a qualified controlled N=8 point: two-scale collision-compatible basis, incremental structured Fourier compression without raw bulk tensors, globally audited MPO bonds, native AD, and independent chi-16 local optimization | Repair chi-32 local-solver intermediates, close the auxiliary raw-gradient audit with a gauge-aware control, and strengthen N=8 basis/reference convergence before N=10 or asymptotic claims |

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
now extended by Gate F to one controlled `N=8,D=10` point. Incremental
structured compression removes the temporary dense raw Fourier-MPO blocks, and
the two-scale basis reduces the N=6 reference discrepancy by `81.5%`. The
constant-in-N Fourier channel count still does not establish continuum or
end-to-end asymptotic scaling: N=8 basis/reference error grows at fixed local
order, the strict auxiliary raw-gradient threshold misses, and chi-32 local
DMRG exceeds the current GPU memory budget. The decision is numerical evidence,
not a method-priority claim.
