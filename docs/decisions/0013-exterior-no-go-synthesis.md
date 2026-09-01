# ADR 0013: Consolidate the exact exterior corridor as a no-go hierarchy

- Status: accepted
- Date: 2026-09-01
- Primary-proof choice superseded by: ADR 0016 (fixed-bond direct Cayley
  reduction); the two-axis hierarchy remains accepted.

## Context

The project has accumulated two representation theorems and four generations
of exact pair-memory gates. Their logical roles were becoming easy to conflate:
ordinary particle-TT rank is not exterior contraction cost; fixed `Mat_2`
pair powers are not row-ordered `Mat_2` determinants; formal Waring rank is not
physical LC-AGP rank after exterior quotienting; and sparse virtual propagation
is not physical coefficient extraction.

Phase 23 audits these statements against their exact certificates and primary
complexity/APG sources. It also finds that the Phase 22 bandwidth-one APG
permanent reduction gives a shorter generic exact squared-norm obstruction than
the earlier tagged Cayley construction.

## Decision

Adopt the two-axis no-go hierarchy as the current exact FEMPS conclusion:

1. ordinary particle-site TT has an unavoidable binomial exchange rank floor,
   flat and approximation-resistant already for a Slater determinant;
2. exteriorization removes that representation floor but does not make generic
   exact contraction polynomial;
3. bounded coefficient algebras and the tested fixed-state graded memories are
   tractable only through polynomial LC-AGP collapse; and
4. unrestricted order tags and even a bandwidth-one APG path contain exact
   permanent-hard squared-norm instances.

Use the sparse APG reduction as the primary generic contraction proof. Retain
the tagged Cayley reduction as an independent theorem about growing
noncommutative order memory.

Do not claim a universal no-go for every structured or approximate FEMPS.

## Consequences

- Stop opening new generic exact coefficient-algebra solver branches without a
  predeclared structure that excludes both known permanent embeddings.
- Keep finite Pfaffian/LC-AGP solvers as prior-art controls, not the central
  method claim.
- Retain the ordered continuous route as the validated numerical control with
  Hong/Li--Waintal parentage.
- Put any approximate exterior route behind a new Gate K requiring an explicit
  failure probability and non-asymptotic observable/energy error certificate.
- Preserve all exact certificates and theorem drafts as the reproducible no-go
  package pending external mathematical review.
