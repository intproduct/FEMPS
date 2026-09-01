# Single-manuscript publication scope

## Decision

The project currently has one publication manuscript:

- authoritative source: `math/femps_no_go_manuscript.tex`;
- generated review PDF: `output/pdf/femps_combined_manuscript_v4.pdf`.

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
maximum one-form bond `chi=2`. The proved exact squared-norm Turing reduction
adds a bond-one scalar reference by direct sum, so its maximum bond is three.
Therefore:

- hard pointwise amplitude at `chi=2`: proved;
- exact squared-norm hardness for fixed `chi<=3`: proved conditionally on the
  cited Cayley-determinant hardness result;
- exact squared-norm hardness restricted to `chi=2`: conjecture, not theorem.

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

Until then, additional NOCI-equivalent convergence or seed studies may support
the combined paper's numerical audit, and non-NOCI algorithm experiments may
test the gate, but neither may create a title, abstract, outline, or submission
source for a second paper.
