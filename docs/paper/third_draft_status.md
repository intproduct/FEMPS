# FEMPS combined structural/no-go manuscript status

## Status

The third-draft structural checkpoint was completed on 2026-09-01. ADR 0028
supersedes the later two-paper split on 2026-09-02. The authoritative source is
`math/femps_no_go_manuscript.tex`; the current combined review artifact is
`output/pdf/femps_combined_manuscript_v4.pdf`. The previous v3 PDF is retained
as a historical checkpoint.

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
  results. The hard pointwise amplitude at `chi=2` is separated from the proved
  exact squared-norm result at maximum bond three; the `chi=2` norm sharpening
  is explicitly labeled a conjecture.
- The finite functional-basis space is stated to be exactly
  `Lambda^N V_D`, shared with same-orbital FCI and quantum-chemistry DMRG; no
  larger variational space or efficiency advantage is claimed.
- A constructive strictly increasing-channel embedding of scalar AGP and
  blocked odd-particle AGP into one-form FEMPS is proved with bond at most
  `floor(D/2)`, while the possible `O(chi D)` cost of splitting a general
  matrix-valued pair core is kept separate.
- The odd/even discussion now distinguishes parity from site dependence,
  homogeneity, virtual width, and coefficient-algebra closure.
- Pointwise functional identities are separated from any unproved
  polynomial-bit functional-input complexity transfer.
- Exact hardness is separated from relative and additive approximation, and
  energy certification retains the positive denominator-interval condition.
- The bounded Wedderburn--radical collapse now has a detailed appendix proof.
- The interacting diagonal-path FEMPS result is included only as numerical
  evidence for a restricted route, with errors, variance, convergence, and
  antisymmetry residuals; no generic scalability or superiority claim is made.
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

- PDFLaTeX and BibTeX completed successfully for the combined v4 source.
- Current PDF: 14 A4 pages, 368,760 bytes.
- Current PDF SHA-256:
  `C6C1F5F9DE65C47E9AFCFB6A90CC2A5DEAE47AD1A7C01C0A43CC07BE3A9EDECE`.
- Build log: no undefined citation/reference, overfull/underfull box, package,
  LaTeX, or font warnings.
- All 14 pages were rendered to PNG and inspected for clipping, overlap,
  table legibility, formula placement, section transitions, and pagination.
- Standard repository suite: `277 passed, 1 known latticeTN report-path
  warning` in 688.51 s, using a workspace-local pytest temporary directory.

The next project milestone is the preregistered Phase 40 `N=2` soft-Coulomb
explicit-correlation differentiator experiment. It is algorithm research, not
a second-paper project, and must preserve the established truth, variance,
symmetry, memory, independent-axis, and comparator requirements.
