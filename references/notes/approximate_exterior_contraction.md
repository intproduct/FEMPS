# Approximate exterior contraction: source audit

## Question

Phase 22 proves that an upper-bidiagonal, bandwidth-one matrix-pair state can
encode `perm(A)` in its only top exterior coefficient.  Exact hardness alone
does not decide whether a randomized or deterministic approximation is useful
for a variational FEMPS calculation.  The relevant distinctions are the
matrix promise, additive versus multiplicative error, and control of the
Rayleigh-quotient denominator.

## Permanent approximation regimes

### Entrywise nonnegative matrices

Jerrum--Sinclair--Vigoda give a fully polynomial randomized approximation
scheme for the permanent of an arbitrary matrix with nonnegative entries
[@JerrumSinclairVigoda2004PermanentFPRAS].  Therefore the Phase 22 exact
0--1 reduction cannot be quoted as an approximation obstruction.  In the
paired APG embedding, this theorem covers the promise

```text
A_(i,j) >= 0 in one declared paired-orbital gauge.
```

This is a useful positive control, but it is not a generic physical property:
real and complex geminal coefficients have signs/phases, and entrywise
nonnegativity is not preserved by general orbital or geminal changes of basis.

### Hermitian/real positive semidefinite matrices

Positive semidefinite is not the same promise as entrywise nonnegative.
Meiburg proves that no PTAS exists for Hermitian PSD permanents unless `P=NP`,
and no randomized PRAS exists unless `RP=NP`; the result also holds for purely
real PSD matrices [@Meiburg2023PSDPermanent].  The paper in fact excludes
subexponential approximation factors on its hard family.  Ebrahimnejad,
Nagda, and Oveis Gharan subsequently prove exponential-factor NP-hardness and
give an improved simply exponential deterministic approximation algorithm
[@EbrahimnejadNagdaOveisGharan2025PSDPermanent].

This negative subclass is admitted by the Phase 22 construction: its coefficient
array `A` is arbitrary, so it may be chosen real symmetric PSD, including PSD
matrices with negative entries.  The exterior input remains real, sparse in
virtual space, and polynomial in size.

### Arbitrary signed or complex matrices

Aaronson--Hance summarize and extend Gurvits's randomized estimator.  For an
arbitrary complex `M x M` matrix it gives additive error of scale

```text
epsilon ||A||^M
```

in `O(M^2/epsilon^2)` time [@AaronsonHance2014Gurvits].  This is an important
polynomial upper bound, but it is not a relative guarantee.  For cancelling
inputs, `|perm(A)| / ||A||^M` can be zero or exponentially small, so the
additive estimate need not even certify a nonzero state norm.  Squaring a
noisy permanent estimate also introduces bias unless a separate unbiased
construction and variance analysis are supplied.

## APG and tensor-network routes

The Phase 22 path state is APG.  Fischer decomposition gives an exact
`2^(M-1)`-term AGP expansion, and low-rank/selected decompositions are already
studied as approximations to APG [@KawasakiNakatani2024LowRankAPG].  Pairing-
scheme selection and permanent-valued APG coefficients are likewise existing
geminal methods [@RicherKimAyers2025GraphicalGeminals].  A selected expansion
may work empirically, but it must report the discarded exterior-state norm or
observable error; selected term count alone is not a certificate.

Ferris's tensor-network Monte Carlo construction is explicitly unbiased for
the tensor-network quantities it samples [@Ferris2015TNMC].  Applicability to
matrix-valued exterior multiplication is not automatic, and unbiasedness alone
does not bound variance or the ratio of correlated Hamiltonian and norm
estimators.  The generic fermion sign-problem complexity result of
Troyer--Wiese is therefore retained only as a comparator
[@TroyerWiese2005SignProblem].  The project-specific negative result is the
direct PSD-permanent transfer, not an appeal to a generic sign problem.

## Audit conclusion

Approximation splits the candidate rather than rescuing it generically:

1. an entrywise-nonnegative paired coefficient cone has an FPRAS;
2. the admitted real-PSD cone rules out a generic relative-error PRAS under
   the standard `RP != NP` assumption; and
3. arbitrary signed/complex inputs have polynomial additive estimators, but a
   useful energy certificate additionally needs an explicit norm lower bound
   and Hamiltonian-numerator error control.

Any future approximate FEMPS proposal must state which promise excludes the
PSD hard family, how that promise behaves under optimization/gauge changes,
and how estimator errors produce a confidence interval for the variational
energy.
