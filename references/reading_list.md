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

No primary source or repository artifact found so far uses the project's exact
notation `mu_4(m)` or supplies a 16-dimensional rank-22/23 candidate. The
definition in `math/four_forms/README.md` is therefore explicitly a working
reconstruction. The exterior-Gorenstein Hilbert-function and sharp low-rank
quadric/orbit sources remain open audit items.
