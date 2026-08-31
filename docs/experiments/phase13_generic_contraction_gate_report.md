# Phase 13 report: beyond-LC-AGP structure and generic contraction gate

## Outcome

Gate B is `FAIL` for unrestricted dense matrix-wedge FEMPS as an exact solver,
conditional on the standard permanent-complexity assumption. This is a
positive research outcome: the former absence of a contraction formula is
replaced by an explicit polynomial-size hardness reduction, so generic solver
engineering can stop for a mathematical reason.

## Exact LC-AGP relation

For even `N=2M`, every Slater determinant is one AGP:

`(sum_j u_(2j-1) wedge u_(2j))^M / M! = u_1 wedge ... wedge u_(2M)`.

Therefore every matrix-wedge FEMPS has an exact pathwise LC-AGP expansion, and
every finite-basis even exterior state is LC-AGP with at most `binom(D,N)`
terms. At fixed N this is polynomial in D, so “not an LC-AGP state” is not a
valid separation. The meaningful question is joint `(N,D,chi)` succinctness
and contraction without enumerating paths or exterior coordinates.

## Minimal noncommuting candidate

The tested candidate is a virtual-matrix-valued pair power

`Psi_M(B;l,r) = l^T (sum_(i<j) B_ij e_i wedge e_j)^M r / M!`.

It reduces to scalar AGP at `chi=1`, and simultaneously diagonalizable
coefficient matrices reduce to at most chi scalar AGPs. The first genuinely
noncommuting case is `N=4,chi=2`, with coefficient

`Q_ijkl = ({B_ij,B_kl} - {B_ik,B_jl} + {B_il,B_jk}) / 2`.

Norm and one-body matrix elements agree between increasing-exterior and full
`D^4` particle-tensor routes. The exact support recurrence is differentiable
but combinatorial, with time

`O(chi^3 sum_m m^2 binom(D,2m))`.

## Tagged Cayley-determinant reduction

For an `n x n` matrix `A` over `Mat_d`, attach shift tags

`B_(x_i,y_j)=E_(i,i+1) tensor A_ij`.

All symmetrized product orders vanish except row order. The unique coefficient
at `D=N=2n` becomes a known phase and `1/n!` times the row-ordered Cayley
determinant. The latter is permanent-hard already for `d=2`
[@ChienHarshaSinclairSrinivasan2011NoncommDet]. A direct-sum interference
argument transfers amplitude evaluation to exact norm evaluation.

The matrix-pair state embeds into the original one-form matrix-wedge FEMPS with
bonds alternating between `chi` and `chi*D`. In the reduction, all declared
dimensions and bonds grow polynomially. Hence a generic exact norm algorithm
polynomial jointly in `(N,D,chi)` would yield a polynomial permanent
algorithm.

This is a conditional algebraic-complexity theorem, not an unconditional
complexity-class separation. It does not rule out restricted algebras or
controlled approximation.

## Correlation diagnostic

The phase adds

`mu_1 = N / Tr(gamma^2)`.

It is basis/gauge invariant and equals one for a single Slater. It is recorded
as a one-body correlation multiplicity only, not entanglement and not a
canonical FEMPS bond spectrum.

## Novelty boundary

Finite LC-AGP remains a stable functional-basis control, but AGP-CI and
nonorthogonal AGP literature already establish that state organization
[@UemuraKasamatsuSugino2015AGPCI; @DuttaChenHendersonScuseria2021NonorthogonalAGP;
@KawasakiGaoScuseria2026AGPCI]. Noncommutative Pfaffian/quasideterminant
terminology is also established algebraically
[@ArtamonovGoloubeva2012NoncommutativePfaffian;
@EtingofRetakh1998Quasideterminants]. The new defensible result at this stage is
the FEMPS-specific embedding/no-go chain, pending external proof review.

## Verification

- Full suite: `103 passed`.
- New exact-small-system properties: scalar AGP reduction, N=4 formula,
  diagonal LC-AGP collapse, similarity gauge, AD, two-route observables,
  single-Slater multiplicity, tagged determinant for `n=1,2,3`, and original
  FEMPS embedding.
- Python source, scripts, and tests compile.
- `git diff --check` reports no whitespace errors.

## Decision

ADR 0003 stops unrestricted generic exact-solver development. The next phase
will independently harden the reduction and triage restricted coefficient
algebras against an ordered-sector functional-TN pivot.
