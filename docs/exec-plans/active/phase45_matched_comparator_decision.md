# Active execution plan: Phase 45 matched-comparator decision

## Objective

Respond to the failed Phase 44 error-control gate without running a rescue
FEMPS point. Use only existing Phase 44 states/samples and existing Hamiltonian
artifacts to decide whether a genuinely matched Li--Waintal or same-basis DMRG
comparison can answer a question that finite NOCI cannot.

## Immediate tasks

1. Audit the exact state-space and cost matching available for the existing
   ordered-coordinate/Li--Waintal implementation and same-orbital-basis DMRG.
2. Separate comparisons that merely reproduce same-basis CI from those that
   measure an implementation-level accuracy, memory, stability, or scaling
   tradeoff.
3. Freeze at most one comparator route in a new ADR before any new comparator
   result. Do not run new FEMPS coordinates or ordinary NOCI points.
4. Prepare the Phase 44 clean/external reproduction packet, including source
   hashes, initialization disclosure, selection failure, raw coordinate
   archives, and verifier command.
5. Keep the combined structural/no-go manuscript as the only paper. A method
   paper remains closed unless a separately admitted matched comparison or
   external reproduction supplies the missing evidence.

## Stop rules

- If same-basis DMRG is only an alternate eigensolver for the already reported
  CI numbers at these sizes, record that and do not market it as a new physical
  comparator.
- If a Li--Waintal comparison cannot be matched in model, error control, and
  resource accounting, do not run it merely to add a table.
- Do not reinterpret the Phase 44 confirmation subgate as an overall pass.
- Do not change the Phase 44 SE/ESS thresholds or append same-point samples.
