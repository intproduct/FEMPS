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
| Ordered-sector particle MPS removes exchange multiplicity on the controlled N4,D8 grid | exact fixed-grid comparison: ordered ranks `(5,9,5)` versus antisymmetric `(8,28,8)`, with signed reconstruction and latticeTN AD | Distance-coordinate native MPO and scaling gate remain open |

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

ADR 0004 therefore makes ordered-sector/interparticle-distance functional TN
the primary numerical research route. The current latticeTN comparator is an
exact dense truth interface, not yet a production contraction.
