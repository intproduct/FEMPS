# Reading list

## Direct baseline

- Hong et al. (2022), functional tensor networks in a continuous local basis.
  Local copy: `refs/2201.12823v1.pdf`; note: `notes/2201_FTN.md`.

## Closest fermionic representations

- Li and Waintal (2026), first-quantized MPS.
- Li and Chan (2016), Hilbert-space MPS.
- Grassmann tensor-network review/local paper.
- Structure-preserving antisymmetric tensor approximation.
- Continuous sums of Slater determinants.
- AGP reduced-density formulas and nonorthogonal AGP/AGP-CI.
- Number-projected BCS and Pfaffian HFB overlap formulas.
- Quantum-number-projected Pfaffian/mVMC optimization.
- Fermionic particle-RDM entropy and Slater-extremality bounds.

Phase 26 added representative symmetry-adapted TN, graded/Grassmann TN,
AGP/Pfaffian, number projection, mVMC, and fermionic particle-entanglement
sources. Phase 27 must add the primary alternating-four-form, exterior
Gorenstein Hilbert-function, orbit-classification, Lefschetz, and Grassmannian-
secant sources needed to define and audit `mu_4(m)` and the 16D 22/23 branch.

## Phase 27 four-form audit

- De Poi--Faenzi--Mezzetti--Ranestad (2017), especially Section 2 and
  Definition 2.4: identifies `Lambda^4(V*)` with the quadrics in the Pluecker
  ideal of `G(2,V)` and defines the `j`-rank by the contraction map. This is the
  current primary source for the project's `2`-rank convention.
- Cardinali--Giuzzi--Pasini (2017): general geometric framework for alternating
  `k`-linear forms and their radicals/upper radicals. It is relevant to support
  degeneracy but does not by itself define or solve `mu_4(m)`.
- Arrondo--Bernardi--Macias Marques--Mourrain (2019): skew apolarity and
  Grassmann/Skew decomposition rank. Its principal algorithmic range is degree
  at most three. This rank notion is a comparator, not the middle contraction
  rank used in the 16D branch.
- Suciu (2020), especially Sections 4.1--4.3: supplies the graded
  skew-commutative Poincare-duality definition and the alternating top form of
  a PD algebra. Its form/algebra bijection is proved in formal dimension three,
  not four. Phase 27 therefore proves the needed four-form exterior apolar
  quotient directly instead of extending that bijection silently.
- Cohen--Helminck (1988), Theorem 2.1 and Table 1: exhausts the nine nonzero
  `GL(7)` orbits of trivectors over an algebraically closed field. Coordinate
  volume duality plus exact elimination closes
  `mu_4^Q(7)=mu_4^Qbar(7)=12`; the source supplies orbit coverage and the
  independent certificate supplies the rank table.
- Antonyan--Oeding (2022), especially the explicit Cartan subspace and Table
  10: supplies the semisimple normal-form slice and all 94 nilpotent
  `SL(8,C)` normal forms for four-vectors. Exact Cartan and orbit certificates
  close `mu_4^C(8)=mu_4^Qbar(8)=mu_4^Q(8)=12`; this is a project derivation
  from the established classification, not a new classification.
- Galitski--Timashev (1999), Theorem 2.1: for theta-group representations, the
  semisimple part is the unique closed orbit in the orbit closure and every
  semisimple element is conjugate into a Cartan subspace. This is the required
  bridge from the exact Cartan bound to arbitrary four-forms.
- Migliore--Zanello (2017), Theorem 3.2: classifies the analogous-looking
  Hilbert vectors `(1,r,h_2,r,1)` for ordinary commutative Artinian Gorenstein
  quotients through `r<=17`. In particular its `r=16` minimum is 15, so it is
  not the origin of the exterior 16D alternatives 22/23. Symmetric-polynomial
  Macaulay bounds must not be imported into the exterior problem.

No primary source or repository artifact found so far uses the project's exact
notation `mu_4(m)` or supplies a 16-dimensional rank-22/23 candidate. The
definition in `math/four_forms/README.md` is therefore explicitly a working
reconstruction. The exterior apolar perfect-pairing statement is now proved
self-containedly, and the seven- and eight-dimensional values are closed, but
sharp four-form rank strata in dimensions nine and above and the 16D origin
remain open audit items.
