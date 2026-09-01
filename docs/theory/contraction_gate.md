# Gate A contraction analysis

> **Superseded status (2026-09-01).** The route costs and small-system
> identities below remain valid, but the original `OPEN` decision was replaced
> by ADR 0003 after Phase 13 proved a conditional #P-hardness obstruction for
> the unrestricted family. Phases 20--22 then classified several restricted
> algebras and sparse growing-memory candidates. See
> `exterior_no_go_hierarchy.md` for the current boundary.

## Current exact routes

Let the open bonds be `chi_0=chi_N=1`, and write
`P=product_{j=1}^{N-1} chi_j` for the number of unrestricted virtual paths.

### 1. Full particle tensor

Materializing the state in `V_D^{tensor N}` needs `D^N` complex coefficients
before using antisymmetry. It is a useful independent oracle and interfaces
directly with arbitrary one-/two-body operators, but is exponential in `N`.
The present reference antisymmetrizer additionally enumerates permutations and
is deliberately not performance code.

### 2. Determinant path sum

Each virtual path is one decomposable wedge. The norm is

\[
 \langle\Psi|\Psi\rangle
 =\sum_{a,b=1}^{P}\det(U_a^\dagger U_b).
\]

Building one overlap and determinant costs `O(N^2 D+N^3)`, giving

\[
 T_{\mathrm{path,norm}}
 =O\!\left(P^2(N^2D+N^3)\right),\qquad
 M=O(PND)
\]

if all path orbitals are cached. Generalized first- and second-order overlap
cofactors give exact one- and two-body matrix elements, but do not remove the
`P^2` factor. The current minor-by-minor reference is intentionally slower than
an adjugate/compound-matrix implementation.

### 3. Exterior-coordinate dynamic program

After `p` sites, store a vector in
`F^{chi_p} tensor Lambda^p V_D`. Adding one core removes virtual-path
enumeration. In the increasing-index exterior basis the recurrence deletes one
index from each target `(p+1)`-subset with the appropriate shuffle sign.

The exact scalar multiply-add count of the implemented recurrence is

\[
 T_{\mathrm{ext}}
 =\chi_1D+\sum_{p=2}^{N}
 \chi_{p-1}\chi_p\,p{D\choose p},
\]

and peak coefficient storage is

\[
 M_{\mathrm{ext}}=
 \max_{1\le p\le N}\chi_p{D\choose p}.
\]

The final norm is the Euclidean norm of the `Lambda^N V_D` coefficient vector.
This route is polynomial in the displayed bond dimensions and avoids `D^N`,
but it is not polynomial jointly in variable `N` and `D`: near half filling,
the exterior dimension is exponential in `D`.

## Validated operator identities

For path Slaters `U_a,U_b` with overlap `S=U_a^dagger U_b`, the one-body
matrix element is a sum of first cofactors of `S` times
`U_a^dagger h U_b`. A two-body matrix element is analogously a sum of
second-order cofactors times pair-wedge matrix elements. Both formulas are
implemented without assuming an invertible overlap, and agree with full-tensor
application on deterministic complex tests.

Three norm implementations and both operator pairs agree in forward values.
Reverse-mode gradients with respect to every complex core also agree. These
are exact-algebra validations, not scaling results.

## Polynomial subclasses already identified

The diagonal-path embedding of an `R`-term Slater sum has only `P=R` allowed
paths rather than `R^(N-1)`. Its norm and Slater--Condon matrix elements cost
polynomial time, beginning with `O(R^2(N^2D+N^3))` for the norm. `chi=1` is the
single-Slater limit. More generally, any core constraint that keeps the number
of admissible paths polynomial inherits this property.

This subclass is physically meaningful and systematically improvable in
Slater-sum length, but by itself does not establish a generic matrix-wedge
FEMPS advantage over determinant expansions.

## Gate status

**FAIL for unrestricted generic exact contraction, conditional on the standard
permanent-complexity assumption.** The two direct exact routes remain:

- exponential virtual-path count, or
- combinatorial exterior-space dimension.

Phase 13 additionally embeds a row-ordered Cayley determinant over `Mat_2` into
a polynomial-size matrix-pair state and then into one-form FEMPS. Exact squared-
norm evaluation would therefore compute a permanent. Phase 22 gives an
independent obstruction already for an upper-bidiagonal unique-path APG state.

A later structured family may pass only after a fresh gate proves both joint-
polynomial observable contraction and a novelty boundary beyond polynomial
LC-AGP/Gaussian prior art. The current generic FAIL does not rule out controlled
approximation, statistics-carrier factorization, or ordered-sector methods.

## Literature boundary

Fermionic operator-circuit rules show that graded signs can be handled with
only marginal overhead relative to the corresponding ordinary tensor network;
they do not reduce the underlying contraction width. Gaussian/matchgate tensor
networks are different: Wick closure lets contractions update covariance or
generating matrices in polynomial time. Generic matrix-wedge FEMPS has no such
closure in the current definition.

This makes a Gaussian/Pfaffian or bounded non-Gaussian extension the leading
structured-subclass candidate. For fixed particle number, the pure
number-conserving Gaussian limit is already a single Slater (`chi=1`), so any
nontrivial candidate must state precisely what additional projected-pair or
correlation structure it retains.
