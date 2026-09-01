# Phase 28 FEMPS algorithm feasibility audit

## Outcome

The repository is ready for an algorithm-recovery implementation rather than
another representation or rank-classification phase. The selected primary
route is exact diagonal-path FEMPS; a generic stochastic contraction remains a
single gated backup. This is a scope decision, not a numerical claim that the
primary route will outperform CI, AGP, or DMRG.

## What already exists

- Continuous harmonic, Dirichlet-sine, odd-Hermite, and multiscale functional
  bases with one-particle operator matrices.
- Harmonic-pair and factorized soft-Coulomb two-body operators, including
  quadrature/factorization diagnostics.
- Matrix-wedge FEMPS cores, `K=1` Slater and diagonal-path Slater-sum embeddings,
  explicit path and exterior materialization truth routes, and antisymmetry
  residuals.
- Polynomial fixed-number AGP/Pfaffian contractions, finite-sum transition
  matrices, AD optimization, overlap conditioning, deterministic seeds,
  checkpoint/resume, and GPU tests.
- Exact exterior diagonalization/reference paths for bounded `binom(D,N)` and
  ordered-sector controls through the prior resource gates.
- Stable benchmark record utilities containing energy errors, conditioning,
  wall time, peak GPU memory, and antisymmetry fields.

## Feasibility boundary

| Calculation | Current status | Allowed role |
|---|---|---|
| Small full particle/exterior materialization | exact, exponential/combinatorial | truth oracle only |
| Generic matrix-wedge virtual-path sum | exact, exponential in path count | tiny cross-check only |
| Generic exterior dynamic program | exact, central `binom(D,p)` sectors | small-system cross-check only |
| Generic fixed-small-bond exact squared norm | conditionally `#P`-hard | no default solver claim |
| Generic relative squared-norm PRAS | conditionally obstructed on the admitted PSD subclass | additive/conditioned special cases only |
| Fixed-number AGP and finite sums | polynomial exact with prior art | comparator/backend, not renamed FEMPS |
| Diagonal-path Slater-sum FEMPS | exact embedding and singular-safe `K^2` transition solver implemented; optimized well-conditioned path pending | Phase 28 primary route |
| Ordered-sector continuous TN | controlled first-quantized comparator | external method control |

## Missing pieces on the primary route

1. An optimized well-conditioned transition path and CPU peak-memory
   instrumentation; the singular-overlap-safe determinant reference is done.
2. A production structured state object that stores `O(KDN)` data rather than
   dense `O(NDK^2)` diagonal cores.
3. Finite-difference gradients; reverse-mode/full-materialization equivalence
   is complete.
4. Energy variance beyond bounded exterior truth and the E3/E4 extensions of
   the existing independent `D`/`K` runner.
5. Completion of the single reproducible E1--E4 ladder in one result schema.

## Route comparison

The exact restricted route wins the first sprint because its estimator error
is zero, antisymmetry is structural, the original FEMPS embedding is explicit,
and every cost can be audited. Its limitation is determinant-count growth and
strong overlap conditioning, both of which the benchmark must measure.

The stochastic generic route could retain more of the matrix-wedge core
expressivity, but amplitude/observable evaluation inherits sign and path
problems. It remains inactive until a non-asymptotic estimator specification is
available. Starting both routes now would reproduce the prior direction drift.

## Immediate acceptance checks

- `K=1` norm/energy equals the independent Slater truth and reports zero
  structural antisymmetry residual.
- Random tiny `K>1` values and gradients agree with explicit exterior
  materialization.
- E1 and E2 reproduce exact/reference energies with deterministic checkpoints.
- E3 shows `K=1` while ordinary particle TT retains the binomial exchange rank.
- E4 supplies the first decisive `D` and `K` convergence table.
