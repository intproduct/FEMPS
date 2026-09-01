# Active execution plan: Phase 27 four-form rank-spectrum reconstruction

## Objective

Resume Workstream A's independent four-form program, with special attention to
the master plan's 16-dimensional rank-22/23 branch. The repository currently
contains no four-form proof artifacts, so this phase begins by reconstructing
the exact definitions, field conventions, prior results, and certificate
provenance before making any extremal claim.

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
  gives `mu_4(7)=12`; dimensions eight and above remain under audit.
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

## Exit criterion

Phase 27 passes only when the four-form problem and base fields are unambiguous,
every claimed value/bound has a clean exact proof or independently verifiable
certificate, chart/orbit coverage metadata are complete, and the 16D 22/23
branch is either resolved or reduced to a precisely documented open case.
