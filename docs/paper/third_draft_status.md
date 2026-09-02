# FEMPS combined structural/no-go manuscript status

## Status

The final Paper A content checkpoint was completed on 2026-09-02. ADR 0028
supersedes the later two-paper split. The authoritative source is
`math/femps_no_go_manuscript.tex`; the canonical frozen review artifact is
`output/pdf/femps_paper_a_frozen.pdf`. Previous PDFs are historical
checkpoints. Paper A is content-frozen and awaits human scientific and wording
review; no further edit is authorized without explicit user feedback.

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

- PDFLaTeX and BibTeX completed successfully for the frozen source.
- Current PDF: 17 A4 pages, 409,039 bytes.
- Current PDF SHA-256:
  `D688D4939A389CA50F7B2F130BD24BFA13F4FD901CC0D026ADEC2C7EDE456FD9`.
- Build log: no undefined citation/reference, overfull/underfull box, package,
  LaTeX, or font warnings.
- All 17 pages were rendered to PNG and inspected for clipping, overlap,
  table legibility, formula placement, section transitions, and pagination.
- The four-cell table was separately compiled and inspected in an APS RevTeX
  two-column full-width sample.
- Standard repository suite: `313 passed, 1 known latticeTN report-path
  warning` in 674.52 s, using the isolated source and a workspace-local pytest
  temporary directory. The exact-certificate subset reports `11 passed`; the
  shifted-Legendre verifier passes orders two through six.
- The required 2201 CPU baseline completes 500 steps with final energy
  `1.8788029184435575`, absolute error `1.0805366005195438e-4`, and elapsed
  time 11.12 s.

The single restored `N=4,D=8` point is complete and retained internally. It
failed the registered final CI-error and variance thresholds; no rescue or
further small NOCI-equivalent point is admitted. The active publication
priority is the combined manuscript's theorem/proof/citation closure. The
explicit-correlation differentiator remains parked and is algorithm research,
not a second-paper project.
