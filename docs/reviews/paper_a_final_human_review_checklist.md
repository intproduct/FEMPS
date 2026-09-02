# Paper A final human-review checklist

## Frozen artifact

- Canonical source: `math/femps_no_go_manuscript.tex`
- Canonical PDF: `output/pdf/femps_paper_a_frozen.pdf`
- PDF: 17 A4 pages, 409,039 bytes
- SHA-256:
  `d688d4939a389ca50f7b2f130bd24bfa13f4fd901cc0d026adec2c7ede456fd9`
- Status: content frozen; awaiting human scientific and wording review; no
  further edit without explicit user feedback.

## Page-by-page review targets

- Page 4: existing fixed-bond squared-norm theorem, unchanged at maximum bond
  two; page 5 records the bond-three generic signed-output polarization only.
- Page 11: Discussion Section 12 begins; P1/P2 definitions, polynomial
  factor-evaluation/bit-complexity base assumption, commutative determinant
  explanation, and division-free ring caveat.
- Page 12: full-width four-cell two-mechanism table; exact shifted-Legendre
  point-evaluation theorem and the first part of its proof.
- Page 13: pointwise proof completion, signed Hamilton-path open cell,
  conditional exact-containment statement, symmetric-Jastrow identity and
  conditional embedding, and the closing future-work sentence.
- Page 17: all references added or newly activated by the final framework.

The four-cell table was also compiled as a full-width float in an APS RevTeX
two-column sample and visually checked for readable type, wrapping, and rules.

## Pointwise theorem checks

- Uses the unnormalized shifted basis `ell_r(t)=P_r(2t-1)` and nodes
  `t_i=i/(N+1)`.
- Gives the integer coefficient formula, nonzero central-binomial-scaled
  Vandermonde determinant, and polynomial bit bounds for evaluations,
  determinant, reciprocal, and inverse.
- Maps CHSS matrices directly to functional-basis coefficients and proves
  `Psi(t)=det(B) 4^(3m) #SAT(phi)/sqrt(N!)`.
- Defines the exact output as `(q,N)` representing `q/sqrt(N!)`, equivalently
  the rational unnormalized value, with no hidden algebraic-number oracle.
- States a polynomial-time metric reduction at maximum bond two and `D=N`.
- Explicitly says the norm and pointwise theorems are independent reductions
  from the same Cayley source.
- Bounded **exact certificate** passes for `2<=N<=6`; verifier SHA-256:
  `f2ed7656126cdb408a312b97f6ac9a8cbf0f323896ef14919d3c67998456b4aa`.

## Literature additions and boundaries

- Berkowitz (1984): division-free determinant computation over arbitrary
  commutative rings.
- Hutter, arXiv:2007.15298: generalized-Slater representation/universality.
- Pfau et al., arXiv:1909.02487: FermiNet and whole-configuration equivariant
  scalar orbital entries.
- Lin--Goldshlager--Lin, arXiv:2112.03491: explicitly antisymmetrized neural
  layers and factorial generic antisymmetrization motivation.
- Zweig--Bruna, arXiv:2208.03264: Jastrow-versus-finite-Slater expressive
  separation only.
- Mertens--Moore, arXiv:1110.1821: neighboring fermionant/immanant complexity,
  not the signed Hamilton-path polynomial.

The bounded search covered Hamilton path/cycle polynomials, alternating/signed
Hamilton paths, sign imbalance, even-minus-odd Hamilton-path counts,
fermionants, and immanants. No existing classification was found for the exact
weighted path polynomial in Eq. (11); the manuscript makes no absolute
priority claim and leaves the cell open.

## Rejected stronger claims

- The four cells are not a complete dichotomy or classification theorem.
- Universal approximation is not exact polynomial-overhead containment.
- No neural ansatz is declared `#P`-hard without an exact containment lemma.
- `J A(M)=A(JM)` does not transfer ambient FEMPS norm hardness to the
  Slater--Jastrow subclass.
- Zweig--Bruna is not used as an evaluation-complexity theorem.
- Exact worst-case point evaluation does not rule out QMC/VMC, approximation,
  promises, or particular low-bond instances.
- A cycle version is not substituted for the path polynomial; even-`N` cyclic
  cancellation is stated.

## Open items for human review

- Verify CHSS Theorems 3.5/3.9 over `Q`, row order, structured boundary, and
  bit-complexity specialization.
- Independently verify the shifted-Legendre determinant/inverse bounds and the
  `(q,N)` exact-output model.
- Review whether P1/P2 terminology and the placement of FermiNet determinant
  channels are maximally clear.
- Review the bounded-search wording for the signed Hamilton-path cell.
- Review the conditional Jastrow basis-product closure and bond statement.
- Perform final scientific attribution, prose, and APS submission-format
  review. No Paper C or new theorem task is opened by these items.

## Validation record

- Final PDF build: success; no undefined citations/references, duplicate
  labels, overfull/underfull boxes, package warnings, or font warnings.
- Visual review: all 17 pages passed for clipping, overlap, broken glyphs,
  float placement, formulas, references, and pagination.
- Exact-certificate subset: 11 passed in 6.60 s.
- Full suite: 313 passed in 674.52 s; one known latticeTN report-path scalar
  conversion warning.
- 2201 CPU baseline: 500 steps; final energy `1.8788029184435575`; absolute
  error `1.0805366005195438e-4`; elapsed 11.12 s.
