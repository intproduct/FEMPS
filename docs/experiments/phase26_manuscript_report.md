# Phase 26 report: manuscript v1 and fixed-bond proof audit

## Outcome

The project now has one self-contained internal synthesis manuscript for the
ordinary particle-TT representation obstruction and the tested exterior
contraction corridor. The Phase 26 audit also resolves the reviewer's
fixed-small-bond checkpoint with a stronger direct theorem.

## Fixed-bond theorem

For matrix entries `A_ij in Mat_2(Q)`, define the site-indexed one-form core

```text
calA^[i] = sum_j A_ij e_j.
```

The increasing top-form coefficient of
`u^T calA^[1] wedge ... wedge calA^[N] v` is exactly
`u^T CDet(A) v`. Absorbing `u,v` gives maximum bond two. Direct-summing a
scalar reference top form produces coefficient `x+1` at maximum bond three,
so two squared-norm values recover

```text
x = ((x+1)^2 - x^2 - 1)/2.
```

Four standard boundary pairs recover the `2 x 2` Cayley-determinant output.
Therefore exact rational squared-norm contraction for the generic one-form
class with `chi<=3` is #P-hard under polynomial-time Turing reductions,
conditional on the published noncommutative-determinant theorem.

## Independent evidence

The exact-integer verifier imports neither PyTorch nor FEMPS. For every
`2<=N<=6` it checks:

- the direct physical-permutation and virtual-path sum;
- all four standard matrix boundary pairs;
- the explicit `2+1` virtual direct sum;
- signed-amplitude recovery from squared norms; and
- zero antisymmetry residual.

The certificate hash is

```text
1d2208d3e5cb14f5c8e6c875f7fddf51c47ce9a3e61be6cedb8246d662b3a016
```

The production helper constructs the corresponding one-form cores, and
floating-point materialization tests cover orders two through four against an
independent Cayley-determinant implementation.

## Manuscript integration

`math/femps_no_go_manuscript.tex` combines:

- R1--R2: exact and approximate ordinary particle-TT exchange floors;
- C1: the direct fixed-bond exact contraction obstruction;
- C2--C3: bounded-algebra and selected growing-memory LC-AGP collapses;
- C4: the independent bandwidth-one APG permanent obstruction;
- C5: the real-PSD relative approximation boundary and energy certificate;
- C6: the universal statistics-carrier tensor-product obstruction; and
- the attributed 2201 reproduction and ordered COM/gap route as control
  evidence only.

The source includes a proof-audit ledger, exact-certificate table and hashes,
limitations, prior-art boundaries, and an AI-assistance disclosure. The final
11-page build has resolved citations/cross-references and no LaTeX, overfull,
underfull, or undefined-reference warnings. A full contact-sheet inspection
shows no clipping or broken page flow.

## Verification

- Fixed-bond and exact-certificate regression: `22 passed`.
- Complete repository suite: `209 passed` with one unchanged latticeTN
  report-path scalar-conversion warning.
- Unified manuscript: 11 pages, resolved bibliography/cross-references, no
  LaTeX or box warnings after the final build.
- Independent contraction theorem draft: 8 pages and no unresolved or layout
  warnings after two final passes.
- Repository whitespace audit passes apart from platform line-ending notices.

## Scope

The phase does not prove that every exterior ansatz or approximation is hard.
It does not establish a new scalable FEMPS solver, a universal fermionic
entanglement spectrum, or novelty of AGP/APG/Pfaffian and ordered-coordinate
methods. All broad classification proofs still require external mathematical
review.

## Decision

ADR 0016 accepts manuscript v1 for external proof review and opens an
independent four-form/exterior-geometry phase. Generic exact matrix-wedge
optimization remains closed.
