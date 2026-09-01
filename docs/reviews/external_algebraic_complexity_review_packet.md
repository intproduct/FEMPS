# External algebraic-complexity review packet

## Status and requested reviewer

This packet is ready to send to an independent human researcher in algebraic
complexity. No human sign-off has yet been received. An internal independent
AI audit identified the bond-two sharpening; that audit is not represented as
human peer review.

## Claims requiring independent review

1. **Fixed-bond exact squared norm.** Check the use of CHSS Theorems 3.5 and
   3.9, especially the structured output `a I_2+b J_2`, the boundary
   `u=e_1`, `v=e_1+e_2`, nonnegativity, metric-reduction postprocessing, and
   the maximum internal FEMPS bond two.
2. **Rational Legendre point value.** Check the rational nodes, nonsingularity
   and polynomial bit length of the Legendre evaluation inverse, construction
   of the Lagrange functions, normalization convention, and one-query CHSS
   transfer.
3. **Bounded-algebra and graded collapse results.** Check whether each claim is
   algebraic or Turing-complexity, whether the decomposition is input or must be
   computed, the rational interpolation construction, and coefficient bit
   lengths.
4. **Sparse APG reduction.** Check the `0-1` source, even-form commutativity,
   factorial convention, perfect-square recovery, and field/inner-product
   assumptions.

## Files

- `math/femps_no_go_manuscript.tex`
- `docs/theory/chss_reduction_audit.md`
- `docs/theory/rational_legendre_pointwise_hardness.md`
- `docs/theory/exterior_no_go_hierarchy.md`
- `scripts/verify_fixed_bond_contraction_obstruction.py`
- `scripts/verify_rational_legendre_pointwise_reduction.py`

## Questions requiring an explicit answer

- Does CHSS's construction over characteristic zero specialize uniformly to
  rational constant-bit matrices with the stated polynomial size?
- Does the fixed boundary recover exactly `a+b`, with no transpose/order
  mismatch after endpoint absorption?
- Is one exact norm query plus integer square root a valid polynomial-time
  metric reduction in the selected function-output model?
- Are the Legendre evaluation inverse and constructed functional coefficients
  polynomial in Turing bit length, not merely arithmetic-circuit size?
- Does the unnormalized point-evaluation convention avoid every hidden
  algebraic-number operation while remaining a faithful first-quantized
  continuous FEMPS evaluation problem?
- Do any collapse statements currently overclaim polynomial-time construction
  when only polynomial LC--AGP term count has been proved?

## Acceptance record

Reviewer name, affiliation, date, conflicts, requested corrections, and final
sign-off must be recorded here or in a linked immutable report before
submission. Until then, the Legendre transfer remains a conjecture in the
submission manuscript.

Use `docs/reviews/external_human_signoff_template.md` for the named reviewer
record. The separate Phase 44 numerical handoff is in
`docs/reviews/phase44_external_reproduction_packet.md`; neither packet has yet
received external sign-off.
