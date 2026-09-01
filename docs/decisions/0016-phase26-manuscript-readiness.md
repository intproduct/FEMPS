# ADR 0016: Accept the Phase 26 internal manuscript for external proof review

- Status: accepted
- Date: 2026-09-01

## Current-status annotation (2026-09-02)

The publication-scope decision remains historical, but its bond-three boundary
has been superseded by a direct audit of CHSS Theorems 3.5 and 3.9. Their
structured output `a I_2+b J_2`, with `a+b=4^(3m)#SAT`, and the fixed boundary
`u=e_1`, `v=e_1+e_2` prove the exact squared-norm obstruction at maximum
internal bond two. The bond-three polarization below remains valid only for
general signed Cayley outputs. The instruction to open an independent
four-form phase is also superseded by ADR 0017 and the algorithm/physics
recovery priority.

## Context

Phases 1--25 produced two representation theorems, several exact-contraction
obstructions and restricted-algebra collapses, a relative-approximation
boundary, a failed universal carrier factorization, and an independently
validated ordered-coordinate control route. These results had accumulated in
separate theorem drafts and gate reports, with different normalization and
evidence conventions.

The first-draft review also identified a potentially decisive missing point:
whether noncommutative-determinant hardness survives at fixed small FEMPS bond,
rather than only in the Phase 13 growing-tag or Phase 22 growing-width
constructions.

## Decision

The Phase 26 synthesis manuscript is ready for **internal external-proof
review**, not submission.

The fixed-bond checkpoint is resolved affirmatively for the generic one-form
ansatz. Site-indexed bond-two cores have top coefficient equal to a
row-ordered `Mat_2` Cayley determinant. A virtual direct sum with a scalar top
form raises the maximum bond to three and recovers each signed matrix entry
from two exact squared-norm queries. Conditional on the published source
hardness theorem, a generic exact solver for the fixed class `chi<=3` would
imply `FP=#P`.

This theorem is compatible with the fixed-`Mat_2` homogeneous pair-power
collapse. The direct construction retains row order through physical site
labels; a repeated even two-form symmetrizes factor order. The Phase 13 tagged
pair-power and Phase 22 sparse APG reductions remain independent explanatory
mechanisms rather than prerequisites for the fixed-bond consequence.

The unified manuscript records R1--R2 and C1--C6 with explicit evidence labels,
fields, reductions, exact certificate coverage, prior-art attributions, and
limitations. Clean compilation and bounded exact checks establish
reproducibility, not external peer verification.

## External review items

- Check the bit-complexity and oracle accounting of the fixed-bond
  squared-norm polarization reduction.
- Check normalization compatibility between normalized antisymmetric tensors,
  exterior top forms, pair powers, and APG conventions.
- Verify the stated scope of the Chien--Harsha--Sinclair--Srinivasan and
  real-PSD permanent results against the manuscript wording.
- Review the bounded Wedderburn--radical and fixed-state interpolation proofs,
  which are broader draft classifications than their finite certificates.
- Review the exterior cut-rank minor and carrier-divisibility proofs against
  the exterior-algebra and representation-theory literature.
- Complete an independent novelty search before any submission or method-name
  claim.

## Consequences

- Do not resume a generic exact matrix-wedge optimizer: the obstruction now
  holds at fixed maximum bond three.
- Keep Pfaffian/finite-LC-AGP and ordered COM/gap solvers as attributed controls,
  not as a new generic FEMPS method.
- Treat the synthesis source as manuscript v1 and preserve theorem/evidence
  labels until the listed proof reviews are completed.
- Open the next independent mathematical phase on four-forms and exterior
  geometry. Any affirmative family must state a representation map and a
  contraction theorem before numerical optimization.
