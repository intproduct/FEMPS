# Combined manuscript A: theorem, evidence, and citation audit

The publication proof bodies have undergone a statement-preserving
formalization pass.  `PROOF_FORMALIZATION_AUDIT.md` records every replacement
and the four explicit external-dependency flags; it should be read alongside
this higher-level status table.  No theorem statement or evidence level was
changed by that pass.

| Manuscript result | Domain and reduction | Evidence level | External dependency / citation | Remaining review |
|---|---|---|---|---|
| Exact particle-TT bond equals unfolding rank | real/complex finite tensors | theorem | exact TT-SVD; Oseledets (2011) | attribution/priority only |
| Universal alternating exchange floor | real/complex, nonzero `N`-forms | theorem | self-contained diagonal-minor proof | exterior-algebra priority search |
| Flat Slater particle spectrum and truncation floor | orthonormal Slater orbitals | theorem (known result) | Coleman (1963), Eckart--Young | none beyond citation check |
| Direct Cayley coefficient identity | arbitrary field, matrix-valued one-forms | theorem | self-contained exterior/permutation proof | endpoint convention check complete |
| Exact rational squared-norm hardness at maximum bond two | `Q`, CHSS structured family, metric reduction | theorem conditional on published source theorem | CHSS Theorems 3.5 and 3.9 | external human algebraic-complexity sign-off pending |
| General signed Cayley recovery at maximum bond three | `Q`, two norm queries | theorem/remark | self-contained polarization | existing exact certificate covers small generic cases |
| Rational shifted-Legendre exact point-value transfer | rational `ell_n(t)=P_n(2t-1)`, explicit `q/sqrt(N!)` output, `Q`, metric reduction | theorem with exact check through `N=6` | independent CHSS reduction plus Vandermonde and polynomial bit bounds | external human algebraic-complexity sign-off pending |
| Bounded Wedderburn--radical LC--AGP collapse | complex algebraic term bound; rational Turing construction only with decomposition supplied | theorem | Wedderburn--Malcev; Veronese/polarization | verify input-model wording and rational decomposition boundary |
| Fixed-state graded LC--AGP collapse | fixed `w,g`; rational construction with embedding supplied | theorem | Vandermonde interpolation; weighted-automata context | verify inverse bit bounds and quotient lift |
| Sparse bandwidth-one APG norm hardness | zero--one matrices over `Q/R`; complex absolute-square variant | theorem | Valiant permanent theorem; APG prior art | external normalization review |
| Real-PSD relative-norm transfer | stated approximation promise | theorem | PSD permanent hardness source | external approximation-complexity review |
| Energy interval certificate | deterministic inequalities with positive lower norm bound | theorem | self-contained interval arithmetic | none |
| Universal direct carrier tensor product fails | all `N>=3` | theorem | explicit two-Slater `N+2` cut-rank counterexample | representation-theory framing review |
| Diagonal-path numerical illustration | `N=6,D=12,K=4`, 924-dimensional exterior space | numerical evidence | same-basis CI; NOCI literature | no beyond-NOCI claim; initialization disclosed |

## Global claim boundaries

- Exact hardness does not exclude additive, randomized, Monte Carlo, or
  separately promised algorithms.
- The pointwise theorem is a worst-case exact-evaluation statement and does not
  imply that all `chi>=2` FEMPS are unsuitable for QMC/VMC or controlled
  approximation.
- Universal approximation is not exact containment. Hardness transfers to a
  different parameterized class only after an exact polynomial-overhead
  containment lemma.
- The symmetric-Jastrow identity is an embedding statement; no Slater--Jastrow
  norm-hardness result is inferred from hardness of the ambient FEMPS class.
- Occupation-number MPS and second-quantized DMRG remain comparators, not
  FEMPS aliases.
- The selected numerical state is a finite NOCI expansion and supplies no
  independent method novelty.
- The old bond-three theorem remains a correct general polarization statement,
  but is not the sharp structured CHSS norm boundary.
