# FEMPS combined structural/no-go manuscript status

## Status

The third-draft structural checkpoint was completed on 2026-09-01. ADR 0028
supersedes the later two-paper split on 2026-09-02. The authoritative source is
`math/femps_no_go_manuscript.tex`; the current combined review artifact is
`output/pdf/femps_combined_manuscript_v5.pdf`. Previous PDFs are retained as
historical checkpoints.

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
- The unnormalized rational-Legendre functional-input transfer now has an
  internally complete interpolation/bit-complexity proof draft and exact
  small-order checks, but remains a conjecture in submission-facing text until
  external human algebraic-complexity review.
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

- PDFLaTeX and BibTeX completed successfully for the combined v5 source.
- Current PDF: 15 A4 pages, 387,182 bytes.
- Current PDF SHA-256:
  `00B2211664B55471BB093A25CF5C5F0A28A607FFF6774EBFC05F4F3494368EBC`.
- Build log: no undefined citation/reference, overfull/underfull box, package,
  LaTeX, or font warnings.
- All 15 pages were rendered to PNG and inspected for clipping, overlap,
  table legibility, formula placement, section transitions, and pagination.
- Standard repository suite: `287 passed, 1 known latticeTN report-path
  warning` in 624.56 s, using the isolated worktree source and a workspace-local
  pytest temporary directory. The exact-certificate subset separately reports
  `10 passed`, and the rational-Legendre verifier passes orders two through six.

The single restored `N=4,D=8` point is complete and retained internally. It
failed the registered final CI-error and variance thresholds; no rescue or
further small NOCI-equivalent point is admitted. The active publication
priority is the combined manuscript's theorem/proof/citation closure. The
explicit-correlation differentiator remains parked and is algorithm research,
not a second-paper project.
