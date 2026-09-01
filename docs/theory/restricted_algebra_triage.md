# Restricted coefficient-algebra classification after Gates H--J

## Status and scope

This note supersedes the pre-Phase-20 triage. It concerns normalized
matrix-valued pair powers

```text
Psi_M = lambda(Omega^M) / M!,
Omega in A tensor Lambda^2(V),
```

not the row-ordered noncommutative determinant itself. The distinction is
essential: fixed noncommutative matrix blocks can be easy for the symmetrized
pair power even though row-ordered determinants over the same block algebra are
permanent-hard.

A subclass passes the project method gate only if it has exact norm and one-/
factorized-two-body contractions polynomial jointly in all admitted controls,
is systematically improvable, lies beyond polynomial LC-AGP/Gaussian prior
art, and retains the 2201 functional-basis operator interface.

## Current classification

| Coefficient structure | Pair-power result | Complexity/novelty boundary | Gate status |
|---|---|---|---|
| Scalars, simultaneous diagonalization, finite commutative semisimple algebra | finite LC-AGP | established AGP/LC-AGP contraction | control only |
| Fixed `Mat_2(C)` | at most `binom(M+3,3)` AGPs | fixed noncommutativity is erased into a polynomial homogeneous-power expansion | Gate H fail as new family |
| Uniformly bounded largest simple block `p` and radical depth `d` | polynomial LC-AGP by the Wedderburn--radical theorem draft | polynomial at fixed `(p,d)`, but only through established LC-AGP organization | Gate H fail |
| One-generator `C[z]/z^d`, with growing `d` | at most `M(d-1)+1` AGPs | a growing jet remains jointly polynomial LC-AGP | Gate I fail |
| Fixed matrix-state width over a fixed number of truncated commuting counters | explicit polynomial LC-AGP bound | includes the smallest noncommutative alternating-word memory | Gate I fail |
| Growing upper-bidiagonal width with endpoint boundaries | exactly APG; paired-orbital top coefficient is a permanent | bandwidth one and one path still give #P-hard exact squared norm | Gate J fail generically |
| Phase 13 shift-tag algebra | recovers row order inside a symmetrized pair power | #P-hard exact squared norm; radical/path depth grows with order | generic Gate fail |
| Gaussian/matchgate or scalar AGP closure | polynomial covariance/Pfaffian machinery | established Gaussian/AGP state class | backend/control |
| Other growing block, radical, counter, or state-width families | unclassified unless they contain a hard specialization | growth is necessary to evade bounded collapse, never sufficient for tractability or novelty | requires a new gate |

## Why the old `Mat_2` inference was wrong

Chien--Harsha--Sinclair--Srinivasan prove that the **row-ordered Cayley
determinant** over `Mat_2` is as hard as the permanent
[@ChienHarshaSinclairSrinivasan2011NoncommDet]. A fixed `Mat_2` pair power is a
different polynomial: physical two-forms commute, and `lambda(Omega^M)` is a
homogeneous polynomial in only four scalar pair forms. Powers of linear forms
span that polynomial space, giving at most `binom(M+3,3)` AGPs.

Phase 13 becomes hard only after adding a shift register whose depth grows with
the determinant order. Its matrix-unit products annihilate every factor order
except row order. The hard resource is therefore growing order memory, not the
presence of a fixed noncommutative block by itself.

## Easy and hard growing memory

Growth alone also has no definite sign:

- `C[z]/z^d` and fixed-state graded memories grow but collapse by exact
  coefficient interpolation;
- the Phase 22 upper-bidiagonal path grows and is standard APG, whose paired-
  orbital coefficient is a permanent; and
- unrestricted growing shift tags recover the noncommutative determinant.

Consequently there is no dichotomy of the form “bounded is easy, growing is
hard.” The proved statement is narrower: bounded `(p,d)` is LC-AGP; selected
growing families are either LC-AGP or permanent-hard.

## Surviving admissible directions

No exact coefficient-algebra candidate tested through Phase 22 passes both
gates. A future proposal must instead provide one of:

1. a physically constrained growing family whose allowed instances exclude
   both permanent embeddings and polynomial LC-AGP collapse, with a proved
   joint-polynomial observable recurrence;
2. an explicitly approximate contraction with a state- or observable-error
   certificate and exact antisymmetry;
3. a stronger statistics-carrier/correlation-multiplicity factorization with
   a canonical truncation theorem; or
4. the separately validated ordered continuous first-quantized route, stated
   as an integration/control result rather than a new exterior ansatz.
