# FEMPS combined structural/no-go manuscript status

## Status

The final Paper A scientific checkpoint was completed on 2026-09-02. ADR 0028
supersedes the later two-paper split. Explicit user feedback then authorized a
presentation-only rewrite into PRA format and a separate human evidence
companion. The submission source is now `paper/femps_pra_manuscript.tex` and
the review PDF is `output/pdf/femps_pra_manuscript.pdf`. The pre-PRA source
`math/femps_no_go_manuscript.tex` and
`output/pdf/femps_paper_a_frozen.pdf` remain archival checkpoints. The
non-submission audit companion is
`output/pdf/femps_pra_human_evidence_audit.pdf`. Paper A is frozen again and
awaits named human scientific and wording review.

The restricted diagonal-path method draft is now an internal working note, not
a second publication manuscript. Its admitted numerical evidence is summarized
inside the combined paper subject to the NOCI-equivalence limitation. See
`docs/paper/SINGLE_MANUSCRIPT_SCOPE.md`.

## Scientific revisions

- The particle-cut rank floor is retained without a priority claim, while the
  Slater flat particle spectrum is explicitly identified as known.
- Fixed-bond exact squared-norm hardness remains a representation-specific
  algorithm-design constraint, not a claim against all restricted,
  approximate, stochastic, Pfaffian, or ordered-coordinate methods.
- The earlier Theorems 1--3 are again displayed as three explicit structural
  results. A theorem-by-theorem CHSS audit sharpens the exact squared-norm
  obstruction to maximum bond `chi=2`; the maximum-bond-three polarization is
  retained only for arbitrary signed outputs.
- The finite functional-basis space is stated to be exactly
  `Lambda^N V_D`, shared with same-orbital FCI and quantum-chemistry DMRG; no
  larger variational space or efficiency advantage is claimed.
- A constructive strictly increasing-channel embedding of scalar AGP and
  blocked odd-particle AGP into one-form FEMPS is proved with bond at most
  `floor(D/2)`, while the possible `O(chi D)` cost of splitting a general
  matrix-valued pair core is kept separate.
- The odd/even discussion now distinguishes parity from site dependence,
  homogeneity, virtual width, and coefficient-algebra closure.
- Exact point evaluation in the unnormalized rational shifted-Legendre basis is
  now a theorem with an independent CHSS reduction, explicit rational nodes,
  determinant and inverse bit bounds, and a `q/sqrt(N!)` output convention.
  Named-human algebraic-complexity review remains mandatory before submission.
- The Discussion adds a P1/P2 two-mechanism map as an organizing framework,
  not a complete dichotomy. The signed Hamilton-path cell remains open;
  containment and Jastrow consequences are strictly conditional.
- Exact hardness is separated from relative and additive approximation, and
  energy certification retains the positive denominator-interval condition.
- The bounded Wedderburn--radical collapse now has a detailed appendix proof.
- The numerical section now selects only the interacting `N=6,D=12,K=4`
  result in the 924-dimensional exterior space. It reports energy, CI error,
  variance, norm, antisymmetry, time, and memory and discloses its zero-padded
  preoptimized `D=10,K=4` initialization; no beyond-NOCI claim is made.
- Higher-dimensional four-form classification is explicitly parked unless it
  controls an algorithmic or physical decision.
- The diagonal-path solver is identified as NOCI-equivalent numerical evidence,
  not an independent method contribution. No second manuscript is in
  development; a new publication decision is permitted only after an
  independently reproduced non-NOCI explicit-correlation `D`-convergence
  advantage or a matched Li--Waintal/same-basis-DMRG comparison.

## Reviewer and submission hygiene

- Recast the manuscript in APS PRA two-column REVTeX 4.2 format and replaced
  internal audit tables by ordinary scientific discussion.
- Removed all checksum strings, repository commands, certificate inventories,
  and other machine-facing scaffolding from the human publication text.
- Added a separate one-column evidence companion with E1--E9 proof,
  algorithm, numerical-process, result, limitation, reviewer-question, and
  blank sign-off fields. This companion is not a second paper.
- Added or clarified citations for Coleman, Chan--Sharma, Bertsch--Robledo,
  Arvind--Srinivasan, and Chien--Harsha--Sinclair--Srinivasan; retained the
  established AGP, Pfaffian, and variational Monte Carlo prior-art boundary.
- Removed visible `R1/R2`, `C1--C6`, Phase, Gate, and evidence-label wording
  from the submission source.
- Added a named OpenAI Codex/GPT-5-family disclosure with human direction,
  source inspection, output verification, and author responsibility.
- The two reviewer DOCX files were read-only inputs and were neither edited nor
  staged.

## Validation

- PRA PDF: 11 pages in two-column REVTeX format; human evidence companion:
  10 A4 pages in one-column review format.
- Both final PDFs contain no undefined citation/reference, package diagnostic,
  overfull box, checksum, repository command, Phase/Gate label, or placeholder.
- All 21 output pages were rendered and inspected for clipping, overlap,
  equation placement, table legibility, bibliography layout, and sign-off
  spacing.
- The archival pre-PRA PDF remains the previously validated 17-page A4
  checkpoint and was not overwritten.
- Standard repository suite: `316 passed, 1 known latticeTN report-path
  warning` in 675.58 s, using the isolated source and a workspace-local pytest
  temporary directory.
- The required 2201 CPU baseline completes 500 steps with final energy
  `1.8788029184435575`, absolute error `1.0805366005195438e-4`, and elapsed
  time 10.26 s.

The single restored `N=4,D=8` point is complete and retained internally. It
failed the registered final CI-error and variance thresholds; no rescue or
further small NOCI-equivalent point is admitted. The active publication
priority is the combined manuscript's theorem/proof/citation closure. The
explicit-correlation differentiator remains parked and is algorithm research,
not a second-paper project.
