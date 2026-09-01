# FEMPS structural/no-go paper: third-draft status

## Status

Completed on 2026-09-01. The authoritative source is
`math/femps_no_go_manuscript.tex`; the review artifact is
`output/pdf/femps_no_go_manuscript_v3.pdf`.

## Scientific revisions

- The particle-cut rank floor is retained without a priority claim, while the
  Slater flat particle spectrum is explicitly identified as known.
- Fixed-bond exact squared-norm hardness remains a representation-specific
  algorithm-design constraint, not a claim against all restricted,
  approximate, stochastic, Pfaffian, or ordered-coordinate methods.
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

- PDFLaTeX and BibTeX completed successfully.
- Final PDF: 13 A4 pages, 364,782 bytes.
- Final PDF SHA-256:
  `EE08B89F52D87AFFB3DEBBF21247AFE1FB3CEBBCD66DCC727AAA61666D73F4A5`.
- Build log: no undefined citation/reference, overfull/underfull box, package,
  LaTeX, or font warnings.
- All 13 pages were rendered to PNG and inspected for clipping, overlap,
  table legibility, formula placement, section transitions, and pagination.
- Standard repository suite: `238 passed, 1 known latticeTN report-path
  warning` in 22.59 s, using a workspace-local pytest temporary directory.

The next project milestone is the registered nonquadratic soft-Coulomb
transferability benchmark. It must preserve the established truth, variance,
symmetry, memory, and comparator requirements.
