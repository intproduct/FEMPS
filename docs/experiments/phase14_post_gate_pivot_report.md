# Phase 14 report: exact obstruction hardening and post-gate pivot

## Outcome

The generic FEMPS contraction obstruction now has an independent exact
certificate and a paper-level theorem draft. No restricted exterior
coefficient algebra currently passes both joint-polynomial contraction and
novelty. The ordered-sector functional MPS is selected as the primary numerical
research route, without renaming it FEMPS.

## Exact certificate

`math/certificates/verify_tagged_cayley.py` imports neither PyTorch nor FEMPS.
It uses exact sparse integer arithmetic and independently enumerates every
perfect matching and every factor ordering for orders one through four. The
archived certificate hash is

`893077be401414cd810fa1154e618d37d83b58e077732801f2482b3716b2c0c0`.

The verifier reproduces the shift-tagged Cayley-determinant identity for:

- `n=1`: 1 matching, 1 order;
- `n=2`: 3 matchings, 2 orders;
- `n=3`: 15 matchings, 6 orders; and
- `n=4`: 105 matchings, 24 orders.

## Theorem artifact

`math/generic_femps_contraction_obstruction.tex` states the construction over
the rationals, cites characteristic-zero #P-hardness of the `Mat_2` Cayley
determinant, proves the polynomial-bond one-form FEMPS embedding, and gives the
direct-sum norm-interference reduction. It explicitly distinguishes a
conditional complexity theorem from an unconditional class separation.

The local MiKTeX installation has not completed first-run setup, so PDF
compilation remains an environment task. The TeX source and all executable
certificates are present.

## Restricted-algebra result

The classification in `docs/theory/restricted_algebra_triage.md` finds:

- simultaneously diagonalizable/commutative semisimple coefficients are
  explicit LC-AGP;
- commutative nilpotent coefficients are AGP jets with strong border-rank
  overlap;
- upper-triangular/radical-index hierarchies have comparator costs such as
  `poly(N^r)`, failing the joint criterion when r is improvable;
- Gaussian/matchgate closure is established prior structure; and
- a genuine `Mat_2` semisimple sector contains hard instances.

No class is admitted as a new scalable exterior solver.

## Ordered-sector latticeTN comparator

The comparator uses the same `N=4,D=8`, spacing `0.7` harmonic-grid plus
soft-Coulomb Hamiltonian as the controlled ordered truth problem.

| Quantity | Result |
|---|---:|
| Ground energy | `10.550426086401494` |
| latticeTN MPS energy error | `3.55e-15` |
| Ordered Hilbert dimension | `70` |
| Ordered particle-MPS ranks | `(5,9,5)` |
| Full antisymmetric particle-TT ranks | `(8,28,8)` |
| Exact ordered MPS parameters | `800` |
| Native latticeTN norm error | `6.66e-16` |
| Ordered reconstruction max error | `2.92e-16` |
| Signed-extension antisymmetry residual | `0` |
| AD gradients | finite on all four cores |

This is evidence that domain ordering removes explicit exchange multiplicity
on the controlled problem. It is not yet a scaling result: the energy path
still gathers a dense `D**N` tensor, and the ordering/box constraint lacks a
production MPO.

## Verification

- Full suite: `107 passed`.
- Exact certificate verification passed.
- Python compile and whitespace checks passed.
- TeX PDF compile blocked only by local MiKTeX first-run configuration.

## Decision

ADR 0004 selects the ordered-sector/interparticle-distance functional TN for
the next operator-contraction gate. Matrix-wedge FEMPS remains the no-go
theory object; LC-AGP remains a baseline.
