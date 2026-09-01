# Phase 24 report: controlled approximate exterior Gate K

## Outcome

Gate K fails for a generic relative-error squared-norm contraction. The exact
Phase 22 obstruction and the approximation obstruction are now stated as
separate theorems: 0--1 exact permanent hardness alone says nothing about an
FPRAS, while the admitted real-PSD specialization directly transfers published
relative inapproximability to the FEMPS/APG squared norm.

No GPU or automatic-differentiation solver was run. The norm gate fails before
observable implementation, as required by the active plan.

## Approximation classification

| Input promise | Available conclusion | FEMPS consequence |
|---|---|---|
| entrywise nonnegative `A` | Jerrum--Sinclair--Vigoda FPRAS for `perm(A)` | tractable randomized positive cone in a fixed paired gauge |
| real PSD `A`, signs allowed | no PRAS unless `RP=NP`; stronger exponential-factor hardness is known | generic relative squared-norm PRAS is rejected by `M! sqrt(n_tilde)` transfer |
| arbitrary signed/complex `A` | Gurvits-type polynomial additive error | no uniform relative norm or energy guarantee near cancellation |
| selected/low-rank APG or stochastic TN | established heuristic/structured routes | must independently certify state/observable error, variance, and denominator |

Entrywise nonnegative and positive semidefinite are deliberately not conflated.
The signed real positive-definite control

```text
[[1,-1/2],[-1/2,1]]
```

has permanent `5/4` and proves that the PSD promise is not the JSV positive
cone.

## Energy certificate

For simultaneous scalar error bounds and `n_tilde>Delta_n`, Phase 24 derives

```text
|E-E_tilde|
 <= (Delta_h + |E_tilde| Delta_n)/(n_tilde-Delta_n).
```

The corresponding exact confidence interval is obtained from the four ratios
of numerator and positive-denominator interval endpoints. Individual failure
probabilities combine by a union bound; no estimator independence is required.
A ratio of unbiased numerator and norm estimators is generally biased, so
unbiasedness alone cannot satisfy the gate.

## Exact controls

`verify_approximate_exterior_gate.py` uses only integers and rational numbers.
It checks:

- an entrywise-nonnegative matrix;
- exact signed cancellation to the zero exterior state;
- `tau=2^-L` cancellation for `L=4,8,16,32`, where the squared norm is
  `tau^2/4`;
- a signed real positive-definite matrix using exact principal minors; and
- the Rayleigh error bound for positive, negative, and numerator-crossing-zero
  energies.

The certificate hash is

```text
c15e7ff268a962e2790004c7f63d47bedb53be0c887ab0241e671b7fe4ff3b16
```

Existing optimized physical project points are finite LC-AGP/Pfaffian states
with deterministic contraction. They remain positive controls for the
tractable baseline, but they do not test a generic APG estimator. A new
physically optimized K1 run was therefore not admitted after the norm gate
failed.

## Prior-art and novelty result

The FPRAS, additive estimator, PSD inapproximability, APG selection, and
unbiased tensor-network Monte Carlo are all prior art. The project-specific
result is their precise transfer/classification on the sparse FEMPS/APG norm,
together with the Rayleigh denominator contract. The Troyer--Wiese sign-problem
result is cited only as context, not used as the FEMPS proof.

## Verification

- Exact-certificate tests: `7 passed`.
- Complete repository suite: `204 passed` in 16.23 seconds.
- One unchanged latticeTN report-path scalar-conversion warning remains.
- The six-page contraction theorem draft compiles twice with resolved
  citations/references and no overfull, underfull, or undefined warnings.
- `git diff --check` passes apart from platform line-ending notices.

## Decision

ADR 0014 closes generic controlled relative approximation negatively. Promised
positive/additive subclasses remain available only with explicit conditioning
and novelty gates. Phase 25 turns to a statistics-carrier/correlation-
multiplicity factorization, the remaining exterior route named in the master
plan.
