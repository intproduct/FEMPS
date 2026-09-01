# Working paper outline: fermionic functional-TN no-go and alternatives

## Provisional scope

This outline governs the single combined theory/evidence manuscript rather
than a claim of a new scalable generic FEMPS solver. ADR 0028 removes the prior
planned split. Current restricted-solver numerics remain one numerical section;
a separate method paper is not opened without a non-NOCI differentiator.

## 1. Motivation from continuous functional tensor networks

- State the Hong et al. functional-basis/operator/AD construction.
- Pose the unresolved first-quantized fermion extension.
- Separate basis truncation from representation and contraction complexity.

## 2. Why ordinary particle-site TT pays for exchange

- Exact TT rank equals particle unfolding rank.
- Universal `binom(N,k)` alternating rank floor.
- Flat Slater particle-Schmidt spectrum.
- Sharp approximate bond/error corollary.
- Clarify why this does not apply to occupation MPS, ordered sectors, or
  determinant/exterior carriers.

## 3. Exterior matrix-product representation

- Define matrix-wedge one-form cores and gauge action.
- Show strict antisymmetry, `chi=1` Slater limit, and finite Slater-sum
  embedding.
- Present small exact norm/operator oracles only as definition tests.
- State explicitly that compact parameter count is not a contraction theorem.

## 4. Fixed-bond direct Cayley obstruction

- Define site-indexed matrix-valued one-form cores.
- Prove that the top exterior coefficient is the row-ordered `Mat_2` Cayley
  determinant with no order tags and maximum hard-state bond two.
- Recover signed amplitudes from two exact squared norms by adjoining a scalar
  reference block, raising the maximum bond only to three.
- State the fixed-parameter conditional #P-hardness theorem and its N=2--6
  exact certificate.
- State the sharp boundary: hard pointwise amplitude is already at `chi=2`,
  while the proved signed squared-norm reduction has maximum bond three;
  `chi=2` exact-norm hardness remains a conjecture.
- Explain why site-labelled row order does not contradict the homogeneous
  fixed-`Mat_2` pair-power collapse.

## 5. Sparse APG permanent obstruction

- Identify the unique-path upper-bidiagonal state as APG.
- Attribute permanent-valued APG/APIG coefficients.
- Prove the paired-orbital permanent/squared-norm reduction directly from
  Valiant's 0--1 permanent theorem.
- Embed the matrix-pair state into one-form FEMPS.
- Explain why this is contraction hardness, not exterior AGP-rank hardness.

## 6. Independent noncommutative order-memory obstruction

- Introduce matrix-valued pair powers as a subfamily.
- Derive the symmetrized noncommutative Pfaffian coefficient.
- Give the shift-tag Cayley-determinant reduction.
- Embed into one-form FEMPS and reduce amplitude to exact squared norm.
- State the conditional #P-hardness theorem and its exact certificate.

## 7. Restricted-algebra classification

- Prove the `T_2` and fixed `Mat_2` homogeneous-power collapses.
- State the bounded Wedderburn--radical LC-AGP theorem draft.
- Treat the growing `C[z]/z^d` jet and fixed-state graded theorem.
- Emphasize the corrected distinction between row-ordered determinant hardness
  and symmetrized pair powers.

## 8. What remains algorithmically viable

- Polynomial Pfaffian/finite-LC-AGP controls and their direct prior art.
- Split approximate exterior contraction into the entrywise-nonnegative FPRAS
  cone, additive estimators with a certified norm lower bound, and the generic
  relative-norm obstruction inherited from real-PSD permanents.
- Give the simultaneous norm/numerator confidence interval required for a
  certified Rayleigh quotient.
- Reject the literal universal statistics-carrier tensor product by cut-rank
  divisibility, and separate it from valid Hamiltonian-specific symmetry TNs
  and state-adaptive Slater/secant expansions.
- Ordered COM/gap functional MPS as the independently validated control,
  including its Hong/Li--Waintal parentage and finite-scope evidence.

## 9. Numerical and exact evidence

- 2201 baseline reproduction and GPU/backend validation.
- N=2/4/6/8 controls demonstrating the representation distinction.
- Restricted diagonal-path data only as an NOCI-equivalent numerical exercise;
  do not present it as a distinct method contribution.
- Exact certificate table with hashes, fields, and reproduction commands.
- Do not mix exploratory floating-point ranks with mathematical proof.

## 10. Discussion

- Keep all current theory and numerical evidence in one manuscript.
- Require explicit-correlation `D`-convergence advantage or a matched
  Li--Waintal/same-basis-DMRG tradeoff before opening a separate method paper.

- Two independent costs: exchange representation versus exterior contraction.
- Why fermionic signs/Grassmann bookkeeping alone do not create closure.
- Scope limitations: even hard subsequence, selected coefficient-memory
  corridor, and no obstruction for every additive or promised approximation.
- Concrete criteria for any future affirmative FEMPS family.

## Appendices

- A: ordinary particle-TT proofs.
- B: exterior conventions and matrix-wedge algebra.
- C: Phase 13 reduction and bit-complexity details.
- D: Wedderburn/graded interpolation proofs.
- E: sparse-path APG permanent proof.
- F: certificate schemas and independent verifier descriptions.
- G: real-PSD approximation transfer and Rayleigh-quotient error propagation.
- H: statistics-carrier dimension obstruction and symmetry/Slater alternatives.
