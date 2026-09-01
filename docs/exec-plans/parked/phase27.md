# Parked execution plan: Phase 27 four-form rank-spectrum reconstruction

**Status:** parked on 2026-09-01 after the exact eight-dimensional checkpoint.
The unresolved 16D rank-22/23 branch is retained as an open mathematics
problem, not as an active FEMPS algorithm milestone. Phase 28 supersedes this
plan for project priority.

## Objective

Preserve the reconstructed four-form definitions, exact seven- and
eight-dimensional results, field conventions, source transcriptions, and
independent certificates. Do not resume the 16-dimensional rank-22/23 search
unless a later ADR identifies a direct dependency on FEMPS expressivity,
contraction complexity, gauge structure, truncation, or a physics benchmark.

## Checkpoints

- [x] Define `mu_4(m)`, the relevant contraction/catalecticant ranks, concise
  support, orbit equivalence, and “rank 22/23” exactly; record every base field
  and any characteristic exclusions.
- [ ] Audit the primary literature on alternating four-forms, exterior
  Artinian Gorenstein Hilbert functions, orbit classifications, Lefschetz
  behavior, and Grassmannian secants; update the bibliography and novelty
  matrix without relying on secondary summaries.
- [x] Create `math/four_forms/` with a problem statement, convention tests,
  exact-arithmetic utilities, certificate schema, and clean-environment
  reproduction commands separated from production FEMPS code.
- [ ] Reconstruct all known low-dimensional values/bounds and test Hodge dual,
  complementary-cut, support-reduction, and direct-sum consistency identities.
  Dimension seven is now closed exactly over `Q` and its algebraic closure:
  Cohen--Helminck orbit coverage plus an independent rank-table certificate
  gives `mu_4(7)=12`. Dimension eight is also closed over `C`, `Qbar`, and `Q`:
  Antonyan--Oeding's Cartan and 94 nilpotent normal forms, the theta-group
  orbit-closure theorem, and an independent exact certificate give
  `mu_4(8)=12`; dimensions nine and above remain under audit.
- [x] Locate or regenerate the 16D rank-22/23 candidate data; if no provenance
  exists, label it a conjectural target rather than inherited evidence.
- [ ] Choose exact rational and/or recorded finite-field charts and specify how
  rank, orbit/chart coverage, and cross-field lifting will be certified.
- [x] Build an independent verifier that checks certificate hashes and imports
  neither PyTorch nor `femps`.
- [ ] Resolve the 16D branch or state the sharpest certified interval together
  with the exact unresolved obstruction; do not substitute floating-point rank
  experiments for proof.
- [ ] Relate any proved four-form spectrum result back to N=4 physical states
  only after the abstract theorem is closed, keeping the mathematical and
  FEMPS manuscripts logically separable.
- [ ] Issue a Phase 27 theorem/status report and ADR deciding whether the result
  belongs in a separate mathematics manuscript or only as a FEMPS citation.

## Parking boundary

The phase is parked with exact theorem/certificate checkpoints through
dimension eight and the rational 16D upper bound 24. Rank 22/23 remains
unresolved and is not an admitted theorem, certificate, or algorithm claim.
Reactivation requires a new ADR with a concrete algorithmic or physical
dependency; open-ended spectrum completion is insufficient.
