# Restricted coefficient-algebra triage after the generic contraction gate

## Purpose and evidence level

The tagged Cayley-determinant reduction rejects unrestricted dense
matrix-wedge FEMPS as a generic exact polynomial solver. This note asks which
coefficient algebras might evade the reduction. “Known cost” below distinguishes
a theorem for the exact FEMPS norm from an analogy with noncommutative
determinant algorithms; the latter is not promoted to a FEMPS result.

A subclass passes the project gate only if:

1. norm and one-/two-body contractions are polynomial jointly in
   `(N,D,chi)`;
2. increasing its control parameter is systematic without moving that
   parameter into the polynomial exponent;
3. it is not merely a fixed/polynomial LC-AGP list, an AGP tangent/border-rank
   reparameterization, or a standard Gaussian state; and
4. it retains the 2201 first-quantized functional-basis operator interface.

## Classification table

| Coefficient algebra | Algebraic behavior | Contraction status | State-family boundary | Gate status |
|---|---|---|---|---|
| Scalars / `chi=1` | commutative field | proved polynomial AGP/Pfaffian formulas | single AGP | control only |
| Simultaneously diagonalizable matrices | conjugate to a subalgebra of diagonal matrices | proved reduction to at most chi scalar AGPs | explicit LC-AGP | contractible but prior-art baseline |
| Finite-dimensional commutative semisimple algebra over C | direct product of scalar fields | componentwise scalar Pfaffians; boundary functional gives a finite component sum | LC-AGP with at most algebra dimension terms | baseline, no new carrier |
| Commutative local/nilpotent algebra, e.g. `C[epsilon]/epsilon^r` | truncated jets of scalar pair matrices | coefficient arithmetic is polynomial for fixed r; full norm/operator derivative formulas still require a written proof | AGP derivatives/jets; strong border-rank and AGP-CI overlap | research control, novelty unlikely |
| Square-zero extension (`radical^2=0`) | at most one nilpotent insertion survives | expected AGP plus first tangent response; can be derived from AGP transition derivatives | tangent/border-AGP object | reject as central novelty unless a distinct physical theorem appears |
| Bounded radical index r with commutative quotient | at most `r-1` radical insertions | Cayley determinant has `poly(n^r)` algorithms in the cited algebraic setting; no automatic FEMPS norm theorem | bounded-order AGP jet hierarchy | fails joint gate if r is an improvable control in the exponent |
| Full upper-triangular `chi x chi` matrices | radical index chi | determinant comparator scales as `poly(n^chi)` | noncommuting but exponent grows with chi | reject joint-polynomial claim |
| Algebra containing a full `Mat_2` semisimple block | genuinely noncommutative quotient | tagged reduction is #P-hard in characteristic zero | includes hard generic instances | reject exact generic solver |
| Block diagonal repeated fixed noncommuting blocks | contains `Mat_2` once block size is at least two | hardness survives projection to one block | not rescued by total block sparsity | reject exact generic solver |
| Gaussian/matchgate closure | scalar quadratic fermionic algebra/covariance closure | established polynomial Gaussian machinery | Gaussian/free-fermion state or number projection | useful backend/control, not new beyond-LC-AGP carrier |

## Commutative and diagonalizable cases

If

`B_ij = S diag(f_ij^1,...,f_ij^k) S^-1`,

then

`l^T Omega_B^M r / M!`

is exactly a sum of at most k scalar AGPs. More generally, a finite-dimensional
commutative semisimple complex algebra decomposes into scalar components. This
is genuinely polynomial but is precisely explicit LC-AGP organization, whose
state and matrix-element machinery predates this project.

Commutativity without diagonalizability introduces nilpotent jets rather than
generic matrix words. For

`B_ij(epsilon)=sum_(a=0)^(r-1) epsilon^a F_ij^(a)` with `epsilon^r=0`,

the state is a boundary-selected coefficient in the truncated Taylor series of
a scalar AGP with pair matrix `F(epsilon)`. At `r=2` it is an AGP plus a tangent
direction. Higher fixed r gives bounded-order derivatives. These objects may
have efficient derivative contractions, but their proximity to AGP border-rank
and recent compact LC-AGP constructions makes novelty especially weak.

## Why upper triangularity is not enough

Chien et al. give a `poly(n^r)` determinant algorithm when the radical has
nilpotency index r, including `r=chi` for the full upper-triangular algebra.
This establishes an important algebraic comparator, not a norm contraction for
our symmetrized pair state. Even if an analogous FEMPS formula is derived, an
`N^chi` or `N^r` cost is not polynomial jointly in the improvable virtual
control. Keeping r fixed restores a polynomial exponent but fixes the maximum
noncommutative insertion order, so it does not provide the required systematic
path to the generic family.

## Gaussian boundary

Ordinary scalar pair exponentials and their number projections have
determinant/Pfaffian closure. Gaussian fermionic MPS algorithms already exploit
this structure. A proposed restricted FEMPS that can be rewritten entirely in
Gaussian covariance data is therefore contractible but does not supply the
desired beyond-Gaussian correlation multiplicity. Adding arbitrary virtual
matrix coefficients leaves Gaussian closure and re-enters the hardness
obstruction.

## Triage decision

No restricted coefficient algebra currently passes both the joint-polynomial
and novelty gates:

- the proved tractable cases collapse to AGP/LC-AGP or Gaussian structure;
- bounded nilpotent cases look like AGP jets/border rank and put the hierarchy
  order into the cost exponent; and
- a genuine noncommutative semisimple `Mat_2` sector already contains the hard
  reduction.

The bounded-jet family remains useful for a precise lemma or negative boundary,
but it should not delay the ordered-sector functional-TN comparator. Any later
restricted proposal must identify an additional physical constraint not
captured by this algebra-only classification.
