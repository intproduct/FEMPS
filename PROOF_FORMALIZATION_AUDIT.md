# Proof formalization audit

## Scope and invariants

This pass rewrites proof bodies in `paper/femps_pra_manuscript.tex` for
line-by-line human audit. It does not change a theorem, lemma, or corollary
statement; their fourteen environment bodies are byte-for-byte identical to
the pre-pass source after newline normalization. Their order, theorem
numbering, citation keys, novelty boundaries, and numerical claims are also
unchanged.

`AUDIT FLAG` means that a step is imported from a cited source and is not
proved by the manuscript itself. A flag is not a new objection or a repaired
claim. It identifies the exact external fact that a human reviewer must check.

## Theorem-by-theorem record

| Result | Original implicit step | Explicit replacement | Any audit flag | Was new reasoning introduced? |
|---|---|---|---|---|
| Structural result I: exact particle-TT ranks | The cut factorization `C_(k)=L_k R_k` and the simultaneous converse were described without indices. | Defines every TT coefficient and both cut matrices entrywise; derives `rank C_(k) <= chi_k`; gives the successive residual matricization, injective left factor, and exact-rank factorization establishing all cut ranks in one TT. | None. The Oseledets citation retains its original role. | No. This expands the existing TT-SVD/rank-factorization argument. |
| Structural result II: universal exchange floor | “Diagonal up to signs” and “no larger contraction image” were asserted verbally. | Defines the indexed minor `(M_I)_{A,B}`; proves off-diagonal entries vanish through `A\B != empty`; evaluates its determinant; then bounds a decomposable form through `dim Lambda^(N-k) U`. | None. | No. The same minor and contraction-image argument is written explicitly. |
| Strict-antisymmetry truncation corollary | No separate proof environment was present. | Writes the contradiction chain `r_k^TT = rank C_(k) >= binom(N,k)` by direct invocation of Structural results I and II. | None. | No. It is only the already-stated immediate consequence. |
| Structural result III: flat Slater particle spectrum | The partition of permutations, shuffle sign, orthonormality, and Eckart--Young sum were compressed into one paragraph. | Defines `Phi_I`, `Phi_(I^c)`, the shuffle `rho_I`, and its sign; decomposes every permutation uniquely; derives the coefficient `sqrt(k!(N-k)!/N!)`; writes the singular-value tail sum and ceiling condition. | None. The Coleman citation is unchanged. | No. The original permutation partition and Eckart--Young argument are expanded. |
| Direct Cayley coefficient lemma | The proof only said that top-form terms choose a permutation. | Expands the matrix-wedge product over `(j_1,...,j_n)`; eliminates repeated indices; converts each surviving exterior monomial to `sgn(sigma)e_1 wedge ... wedge e_n`; and writes endpoint boundary absorption explicitly. | None. | No. This is the coefficient calculation already described. |
| Fixed-bond squared-norm hardness | The source-to-oracle reduction, boundary multiplication, norm, size, bit length, and postprocessing appeared in one prose block. | Separates source instance, constructed cores, bond audit, oracle value, answer recovery, encoding size, output bit bounds, and the complete metric-reduction chain. | **Yes.** CHSS Theorems 3.5 and 3.9 supply the structured row-ordered output, polynomial construction, field condition, and gadget bounds; standard `#SAT` completeness is also external. | No. The existing CHSS reduction is only made explicit. No missing CHSS fact is derived. |
| Bounded-algebra LC--AGP collapse | Radical filtration, structural choices, commuting polynomial variables, power spanning, and Turing bounds were summarized. | Writes the radical-word decomposition by `k`; defines run lengths and structural label `tau`; gives the variable count `v_k`, simplex size, AGP count, fixed-parameter polynomial bounds, and points to Appendix A for the existing interpolation and bit estimates. | None. The supplied Wedderburn--Malcev decomposition remains an assumption. | No. This reorganizes the existing proof and Appendix A derivation. |
| Fixed-state graded LC--AGP collapse | The quotient lift, interpolation grid, polarization factor, and rational complexity were not individually indexed. | Defines `L_j`, each Vandermonde matrix, the tensor grid size `G`, the `w^2` physical variables, the power count `S`, `K <= GS`, and fixed-`w,g` time/bit bounds; cites Appendix B for the existing inverse estimates. | None. The embedding remains supplied input. | No. This is an indexed expansion of the existing proof and Appendix B. |
| Sparse APG permanent obstruction | The permanent expansion and complexity recovery were stated in three sentences. | Defines the unique-path pair matrix; expands over all label tuples; proves survival iff the tuple is a permutation; uses even-degree commutativity; derives norm, exact recovery, physical dimension, virtual width, input size, output bit length, and the full reduction chain. | **Yes.** Valiant's zero--one permanent hardness theorem is cited but not reproved. | No. The existing permanent reduction is expanded without adding a new reduction. |
| Real-PSD relative-norm transfer | “After a polynomial tolerance conversion” was not shown. | Separates PSD source, sparse-path instance, relative-norm event, square-root relative-error identity, recovered permanent estimate, zero case, size/bond/bit data, and randomized postprocessing complexity. | **Yes.** The precise Meiburg inapproximability theorem and its promise/convention remain external. | No. The tolerance conversion implicit in the original proof is written algebraically. |
| Rayleigh-quotient certificate | The proof cited one identity and the positive denominator. | Derives the norm lower bound, expands `n(E-E_tilde)`, applies the triangle inequality term by term, and states the simultaneous random-event condition. | None. | No. This is the same elementary identity with every inequality shown. |
| Cut-rank divisibility obstruction | Divisibility and independence of the `N+2` contractions were asserted without listing the contractions. | Takes tensor-product dimensions explicitly; lists all contraction families; proves their independence by unique monomial supports, including the `N=3` empty-wedge convention; and writes the contradictory rank formula. | None. | No. The original two-Slater counterexample and support argument are expanded. |
| Explicit AGP embedding lemma | The virtual-channel construction and the `M!` cancellation were verbal. | Expands `Omega_F^M`, proves repeated labels vanish and even pair forms commute, defines every one-form core entry for `M=1` and `M>=2`, establishes the increasing-channel bijection, and audits the bond. | None. | No. This formalizes the construction already stated. |
| Exact rational-polynomial point evaluation | Basis invertibility, output encoding, alternating point expansion, and reduction complexity were present but partially compressed. | Factors the evaluation matrix into Vandermonde and triangular parts; gives denominator clearing and Hadamard bit bounds; expands point evaluation over particle and basis indices; identifies the determinant with repeated/permuted columns; and separates source, constructed instance, oracle output, answer recovery, bond, coefficient bit length, and postprocessing. | **Yes.** The structured CHSS identity and standard `#SAT` classification remain external and are not reconstructed. | No. The same reduction is written with all intermediate maps and formulas. The former punctuation error in the displayed product was treated as transcription, not as a new mathematical step. |

## Audit-flag index

1. **Fixed-bond squared-norm hardness:** external CHSS structured-output and
   source-problem dependency.
2. **Sparse APG permanent obstruction:** external Valiant permanent-hardness
   theorem.
3. **Real-PSD relative-norm transfer:** external Meiburg approximation-hardness
   theorem.
4. **Exact rational-polynomial point evaluation:** the same external CHSS and
   `#SAT` dependency, independently reused by the point-evaluation reduction.

No other proof step was found to require a new assumption, lemma, identity, or
complexity argument beyond material already present in the main text or
Appendices A and B. This statement records the repository-internal
formalization result; it does not substitute for the named human review
requested by the project plan.
