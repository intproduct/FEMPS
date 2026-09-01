# Phase 21 report: fixed-state graded memory collapses

## Decision

Candidate I2 and the broader fixed-state graded candidate class are closed
negatively. The minimal alternating-word algebra is noncommutative, has
dimension `2d-1`, and has radical depth `d`, yet every arbitrary-boundary N=2M
pair state is an exact LC-AGP with

```text
K <= [M(d-1)+1] binom(M+3,3).
```

More generally, fixed matrix-state width `w` and fixed commuting-counter count
`g` give a polynomial term bound. Gate I therefore fails for one-generator and
fixed-state graded growing-memory families. No continuous variational solver is
admitted from these candidates.

## Exact evidence

The standalone verifier compares two independent routes:

- direct multiplication in
  `Q<x,y>/(x^2,y^2,words of length d)`; and
- the nested `Mat_2(Q[z]/z^d)` construction using exact coefficient extraction
  and exact four-variable power interpolation.

Every boundary word passes for all 12 `(M,d)` cases with `1<=M<=3` and
`1<=d<=4`. Certificate hash:

```text
082238ff6e6783b7533b3b2a59f3664beb1820794bcae573add98baf43370030
```

## Prior-art boundary

The finite-state/matrix-linear representation of word weights belongs to
weighted-automata and rational-series theory. Homogeneous power interpolation
belongs to Waring/Veronese theory, and the output state is ordinary LC-AGP. The
project contributes a negative classification at their intersection, not a new
automata, Waring, or AGP ansatz theorem.

## Next boundary

The exact coefficient-algebra search is now confined to growing automaton width,
growing independent noncommutative counters, or other structure outside the
fixed-state graded theorem. Every such candidate must be checked against the
Phase 13 permanent-hard tags before implementation. Approximate contraction is
a separate possible route but must include a quantitative state/observable
error certificate and cannot be inferred from border rank alone.

## Reproduction

```powershell
.\.venv\Scripts\python math\certificates\verify_alternating_word_pair_collapse.py --verify math\certificates\alternating_word_pair_lc_agp_certificate.json
.\.venv\Scripts\python -m pytest -q tests\test_exact_certificates.py
```
