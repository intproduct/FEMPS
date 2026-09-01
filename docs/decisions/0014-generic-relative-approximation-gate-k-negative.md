# ADR 0014: Reject generic relative approximate exterior contraction

- Status: accepted
- Date: 2026-09-01

## Context

Phase 22 proves exact squared-norm hardness by embedding a 0--1 permanent in a
bandwidth-one APG path. Exact hardness does not settle approximation: the
permanent of an entrywise-nonnegative matrix has an FPRAS. Candidate K1
therefore asked whether a randomized or deterministic approximate exterior
contraction could retain exact antisymmetry and provide a non-asymptotic
variational-energy certificate.

The same APG identity accepts arbitrary coefficient arrays. In particular, it
accepts purely real positive-semidefinite arrays, whose permanents are known to
resist randomized polynomial relative approximation unless `RP=NP`.

## Decision

Gate K is **FAIL** for generic Candidate K1.

If every sparse-path squared norm had a relative PRAS, taking its nonnegative
square root and multiplying by `M!` would give a PRAS for every real-PSD
permanent. This contradicts Meiburg's inapproximability theorem unless
`RP=NP`.

Do not infer this result from exact #P-hardness. Preserve the two affirmative
boundaries:

1. entrywise-nonnegative paired coefficients in a fixed gauge admit the
   Jerrum--Sinclair--Vigoda FPRAS; and
2. arbitrary signed/complex coefficients admit Gurvits-type additive
   permanent estimates, but only a declared norm lower bound can convert them
   into a certified Rayleigh quotient.

For scalar estimates satisfying `|n-n_tilde|<=Delta_n`,
`|h-h_tilde|<=Delta_h`, and `n_tilde>Delta_n`, use

```text
|E-E_tilde|
 <= (Delta_h + |E_tilde| Delta_n)/(n_tilde-Delta_n).
```

Unbiasedness without a variance/tail bound and positive denominator interval
is not a Gate K certificate.

## Consequences

- Do not implement a generic approximate matrix-pair GPU/AD solver.
- Any later approximate exterior route must declare a promise excluding the
  real-PSD hard embedding and show that optimization preserves that promise.
- Retain positive-cone, norm-lower-bound, strongly orthogonal, selected-pair,
  Pfaffian/Gaussian, and other structured methods only as explicitly promised
  subclasses with prior-art attribution.
- Advance the exterior program to the remaining master-plan question: whether
  a gauge-independent statistics-carrier/correlation-multiplicity
  factorization exists and yields more than a change of notation.
