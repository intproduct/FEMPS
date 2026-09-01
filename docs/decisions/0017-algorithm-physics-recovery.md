# ADR 0017: Recover FEMPS algorithm and physics as the project main line

- Status: accepted
- Date: 2026-09-01
- Supersedes: ADR 0004 for current method priority; it does not revoke the
  ordered-sector control or any no-go theorem in ADR 0013/0016.

## Context

The project has closed the ordinary particle-TT exchange-rank obstruction,
proved a conditional fixed-small-bond obstruction for generic exact FEMPS
squared norms, built small-system exterior truth engines, and obtained exact
seven- and eight-dimensional four-form checkpoints. Phase 27 then began to
consume the main line with a 16D rank-22/23 problem that does not currently
decide a solver design or physics benchmark.

Generic polynomial exact contraction is no longer a reasonable default success
condition. At the same time, the existing code already supports continuous
functional bases, one-/two-body Hamiltonians, strict exterior antisymmetry,
automatic differentiation, finite AGP/Pfaffian controls, checkpoints, and
small-system references. The missing result is an honestly scoped FEMPS
algorithm/physics closure.

## Decision

1. Park Phase 27 after the exact eight-dimensional checkpoint. Preserve the
   16D interval/open branch without further main-line search.
2. Make Phase 28 Algorithm and Physics Recovery the only active research plan.
3. Select **nonbranching diagonal-path FEMPS** as the primary route. Its global
   conserved virtual label embeds a sum of `K` nonorthogonal Slaters exactly in
   the original matrix-wedge definition. Contract through `K^2` determinant
   transitions, never through all `K^(N-1)` paths.
4. Treat `K` as a structured correlation multiplicity for this subclass, while
   avoiding claims that it is canonical entanglement or solves generic FEMPS.
5. Retain one inactive backup: a controlled stochastic estimator for generic
   cores. It may start only after a separate error/variance/failure-probability
   gate and an antisymmetry-residual definition.
6. Keep finite AGP/Pfaffian, ordinary Slater/CI, ordered-sector first
   quantization, exact diagonalization, ordinary particle TT, and optional
   second-quantized DMRG as explicitly named comparators or backends. Do not
   rename them FEMPS.
7. Run E1--E4 in order. No practical/scalable/advantage claim is admitted until
   a nontrivial interacting benchmark and independent `D`/`K` convergence pass.

## Why this primary route

- It is literally a restricted instance of the current FEMPS state definition,
  so the pivot does not change quantization, particle-coordinate sites, or the
  continuous functional basis.
- `K=1` carries a Slater with correlation bond one; increasing `K` adds genuine
  multideterminant correlation and is systematic.
- Determinant and generalized Slater--Condon transitions provide an exact,
  polynomial, AD-compatible contraction target with explicit complexity.
- The restriction excludes both known permanent embeddings by forbidding
  branching/mixing of the global path label across sites.
- It provides a fast falsifiable experiment. If its determinant count grows too
  quickly or optimization is unstable, that is a useful algorithmic result.

## Consequences

- High-dimensional alternating-form classification is parked unless a new ADR
  identifies a direct solver dependency.
- Existing no-go results become constraints on the allowed structure and claim
  language, not a reason to stop approximate or restricted algorithm work.
- The initial solver may resemble nonorthogonal selected CI. Any claimed FEMPS
  advantage must therefore be demonstrated by exchange/correlation separation,
  functional-basis integration, optimization behavior, or measured complexity,
  not by renaming established determinant technology.
- Failure of the primary and gated backup routes must be reported plainly; it
  cannot be masked by new mathematical classifications.

## Validation update (2026-09-01)

The primary route passed E1--E4 at its registered restricted-algorithm scope.
Three blind `D=6,K=4` starts and three truth-free `D=6 -> D=7` continuations
pass energy, variance, norm, antisymmetry, memory, and no-enumeration criteria.
An independent verifier recomputes the decision from the raw artifact. The
result accepts the route as a useful exact baseline and does not upgrade it to
a generic FEMPS contraction, novelty, scalability, or superiority claim.
