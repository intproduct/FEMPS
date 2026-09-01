# Single-manuscript publication scope

## Decision

The project currently has one publication manuscript:

- authoritative source: `math/femps_no_go_manuscript.tex`;
- generated review PDF: `output/pdf/femps_combined_manuscript_v5.pdf`.

The structural/no-go results and all presently admitted restricted-solver
numerics belong to this manuscript. The existing
`docs/paper/femps_method_manuscript.tex` is a frozen internal working note and
reproduction aid, not a second submission candidate.

## Structural theorem continuity

The three results called Theorems 1--3 in the earlier working package are not
removed:

1. exact ordinary particle-TT rank equals particle-unfolding rank;
2. every nonzero alternating tensor has the universal binomial exchange-rank
   floor, with the strict-antisymmetry truncation corollary;
3. a Slater determinant has the known flat particle-Schmidt spectrum and the
   resulting sharp rank-versus-error formula.

The combined manuscript now presents these explicitly as Structural results
I--III. The prior consolidation of results 1 and 2 into one theorem, and the
prior demotion of result 3 to a proposition, are reversed for discoverability;
their scientific content is unchanged.

## The chi=2 boundary

The direct Cayley construction gives a hard pointwise amplitude already with
maximum one-form bond `chi=2`. CHSS Theorem 3.9 supplies the structured output
`a I_2+b J_2`, `a+b=4^(3m)#SAT`; the fixed boundary `u=e_1`, `v=e_1+e_2`
therefore gives a nonnegative top-form coefficient at the same bond. Therefore:

- hard pointwise amplitude at `chi=2`: proved;
- exact squared-norm hardness at `chi=2`: proved conditionally on the cited
  CHSS structured Cayley-determinant reduction;
- maximum-bond-three scalar-reference polarization: retained only for general
  signed Cayley outputs.

The combined manuscript states this boundary explicitly.

## Status of the restricted solver numerics

The nonbranching diagonal-path implementation is algebraically a finite
nonorthogonal Slater expansion and is therefore NOCI-equivalent. Its present
N2/N4/N6, clean-source, seed, and D/K evidence is retained only as a bounded
numerical exercise and algorithm-design consequence in the combined paper. It
does not establish a new state class or an independent method paper.

## Gate before any new publication decision

A separate method manuscript is not currently planned or drafted. A new
publication decision may be considered only after reproducible evidence closes
at least one genuinely differentiating question:

1. a first-quantized continuous FEMPS carrier beyond finite NOCI supplies
   explicit correlation and a matched functional-basis `D`-convergence
   advantage; or
2. a matched comparison with Li--Waintal and same-orbital-basis DMRG identifies
   a clear accuracy, stability, memory, or complexity tradeoff not already
   explained by an ordinary NOCI expansion.

Until then, no additional small NOCI-equivalent convergence or seed points are
admitted. A future non-NOCI algorithm experiment may test the differentiator,
but it may not create a title, abstract, outline, or submission source for a
second paper before a reproducible result exists.
