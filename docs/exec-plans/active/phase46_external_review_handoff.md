# Active execution plan: Phase 46 external review and reproduction handoff

## Objective

Package the two remaining independent-review obligations without generating
new theorem claims, FEMPS physics points, or manuscript splits:

1. human algebraic-complexity review of the CHSS exact-norm theorem and the
   rational unnormalized-Legendre pointwise reduction; and
2. external clean reproduction of the failed Phase 44 gate, including its
   low-D confirmation subresult and all failed selection diagnostics.

## Deliverables

- one compact theorem-review packet with exact statements, field/encoding
  assumptions, dependency graph, primary citations, certificate commands, and
  explicit conjecture boundaries;
- one Phase 44 reproduction packet with commit IDs, environment, fixture and
  reference-firewall chronology, immutable seeds/budgets, raw archive hashes,
  verifier command, and the required failed-gate wording;
- reviewer checklists that require named human sign-off and do not treat AI
  review as independent human review;
- no edit that promotes Phase 44 to an overall pass before external
  reproduction exists. The user-authorized final Paper A framework patch
  closes the repository-internal shifted-Legendre pointwise proof as a theorem;
  named-human algebraic-complexity sign-off is still required before
  submission and is not manufactured here.

## Prepared handoff checkpoint

- `docs/reviews/external_algebraic_complexity_review_packet.md` identifies the
  exact CHSS/Legendre questions and files.
- `docs/reviews/phase44_external_reproduction_packet.md` records frozen commit
  IDs, clean-run commands, environment/resource expectations, hashes, and the
  mandatory failed-gate language.
- `docs/reviews/external_human_signoff_template.md` separates named human
  theory sign-off from external numerical reproduction and explicitly rejects
  AI or internal self-attestation.
- `scripts/compare_phase44_external_reproduction.py` now authenticates the
  primary artifact, verifies a reproduction from its own checkpoints and raw
  samples, and applies a frozen uncertainty-aware comparison. Its output
  cannot self-attest external independence or authorize Paper B.
- A repository self-check exercises that external-artifact path and passes all
  numerical checks while deliberately retaining
  `external_independent_replication_complete = false`.
- Phase 41's repository-internal manuscript closure is archived as completed,
  and the active-plan index now identifies this Phase 46 file as the sole
  operational plan. The unchanged Phase 40 preregistration path is retained
  only to preserve hashes embedded in frozen evidence.

The repository side of the handoff, including machine-readable result
comparison, is ready. Completion now requires actual external human input;
until received, this phase remains active and Paper A remains frozen for human
scientific and wording review.

## Stop rules

- Do not impersonate or synthesize human approval.
- Do not append new samples or modify Phase 44 thresholds.
- Do not open Paper B.
- Keep the single combined manuscript as the only submission source.
