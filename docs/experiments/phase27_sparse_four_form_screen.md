# Phase 27 sparse four-form screen

## Status and purpose

**Evidence status: numerical evidence.** This experiment is a seeded search for
sparse coordinate candidates in the reconstructed 16-dimensional four-form
problem. It is a diagnostic after the loss of candidate provenance, not a
proof of a lower bound and not orbit/chart coverage.

For a coordinate four-form, each nonzero four-index term contributes three
weighted edges between complementary two-subsets. The middle contraction is
therefore a signed adjacency matrix on active two-subsets. The screen computes
the first and middle ranks exactly over `F_2` using bit elimination, retains
only first-rank-16 samples, and recomputes the best representative of each run
over `Q` by exact rational elimination.

## Reproduction

From the repository root, for `s=5,...,10`:

```powershell
python math/four_forms/explore_hypergraph_forms.py --terms s --samples 20000 --seed (270000+s)
```

The executable command uses literal integers in place of `s` and the seed
expression. No GPU or external package is used.

## Results

| terms | seed | samples | concise over F2 | best F2 middle rank | selected Q rank vector |
|---:|---:|---:|---:|---:|---|
| 5 | 270005 | 20,000 | 19,838 | 24 | `(1,16,24,16,1)` |
| 6 | 270006 | 20,000 | 19,585 | 26 | `(1,16,26,16,1)` |
| 7 | 270007 | 20,000 | 19,589 | 24 | `(1,16,24,16,1)` |
| 8 | 270008 | 20,000 | 19,668 | 24 | `(1,16,24,16,1)` |
| 9 | 270009 | 20,000 | 19,767 | 30 | `(1,16,30,16,1)` |
| 10 | 270010 | 20,000 | 19,870 | 30 | `(1,16,30,16,1)` |

Total: 120,000 samples, of which 118,317 were concise over `F_2`. No
screened unit-coefficient coordinate hypergraph had middle rank below 24.

Because reduction modulo two cannot increase the rank of an integer matrix,
this also rules out rational rank below 24 for the sampled unit-coefficient
forms. It does not rule out:

- other hypergraphs;
- non-unit or non-coordinate rational coefficients;
- candidates whose useful structure degenerates modulo two;
- non-sparse `GL(16)` orbit representatives;
- characteristic-dependent rank phenomena.

## Decision

The search did not recover the lost 22/23 branch. Rank 24 remains the only
certified rational upper bound in the repository, supplied separately by the
direct-sum artifact. The next search must be driven by a structural low-rank
quadric/orbit theorem or a parameterized exact chart, not by treating more
random samples as proof.
