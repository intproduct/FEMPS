# Phase 23 report: exact exterior no-go hierarchy

> Historical phase report. Phase 26 later found a stronger direct Cayley
> reduction at maximum one-form bond three; the sparse APG result remains the
> simplest mechanism using only the ordinary 0--1 permanent.

## Outcome

The project now has one logically consistent theorem/evidence package rather
than a sequence of partially superseded gates. It separates:

- the ordinary particle-TT exchange representation floor;
- generic exact exterior contraction hardness;
- polynomial LC-AGP collapses of bounded/fixed-state memories; and
- the non-universal boundary left for structure or approximation.

The simplest generic exact squared-norm obstruction is now the Phase 22
bandwidth-one APG permanent construction. The Phase 13 tagged Cayley theorem is
retained because it independently shows how growing shift memory restores
noncommutative row order.

## Corrections made

- Replaced the stale Gate A `OPEN` status by the accepted conditional FAIL.
- Corrected Phase 14's provisional claim that a fixed `Mat_2` pair-power sector
  itself contains the hard reduction; Phase 20 proves its `O(M^3)` LC-AGP
  collapse.
- Standardized theorem language on exact **squared norm** `<Psi|Psi>`.
- Prevented ordinary polynomial Waring rank from being quoted as a physical
  LC-AGP lower bound through the exterior quotient.
- Stated that a hard subclass rejects a generic family algorithm, not every
  individual sparse instance.

## Artifacts

- `docs/theory/exterior_no_go_hierarchy.md`: dependency graph, theorem scopes,
  complete certificate hashes, and coverage boundary.
- `docs/paper/no_go_claim_evidence_matrix.md`: admissible and forbidden claim
  language with prior-art dependencies.
- `docs/paper/no_go_paper_outline.md`: manuscript and appendix structure.
- `math/generic_femps_contraction_obstruction.tex`: updated five-page theorem
  draft containing both the tagged and sparse permanent reductions.
- `tests/test_matrix_pair.py`: production exterior/squared-norm/one-form
  regression for the sparse theorem.

## Verification

- Both theorem drafts compile successfully with MiKTeX.
- The updated contraction draft resolves all citations/cross-references and has
  no overfull, underfull, or undefined-reference warnings after two passes.
- Full repository suite: `203 passed` in 14.79 seconds.
- One unchanged latticeTN warning remains in a report-path scalar conversion.
- `git diff --check` passes apart from platform line-ending notices.

## Decision

ADR 0013 closes the tested generic exact coefficient-memory corridor and opens
Phase 24 only as a controlled-approximation feasibility gate. No GPU or AD
solver is admitted from heuristic compression alone.
