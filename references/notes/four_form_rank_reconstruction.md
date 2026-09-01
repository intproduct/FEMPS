# Four-form rank reconstruction (Phase 27)

## Question being reconstructed

For a field `K`, an `m`-dimensional vector space `V`, and
`omega in Lambda^4(V*)`, define

```text
C_j(omega): Lambda^j(V) -> Lambda^(4-j)(V*)
C_j(omega)(x)(y) = omega(x wedge y).
```

The source-backed term is the `j`-rank, `rank C_j(omega)`. The project's
provisional extremal function is

```text
mu_4^K(m) = min rank C_2(omega), subject to rank C_1(omega) = m.
```

The constraint is called concise/full support in project language. The exact
notation `mu_4` has not yet been traced to a primary source, and the master
plan's “16D rank 22/23” text has no candidate or certificate in Git history.
Accordingly this is a **conjectural target**, not inherited evidence.

## Primary-source findings

### De Poi--Faenzi--Mezzetti--Ranestad (2017)

Section 2 identifies four-forms with the quadratic forms in the degree-two
Pluecker ideal of `G(2,V)`. Definition 2.4 defines the `j`-rank of an `i`-form
as the rank of its contraction map. Thus for a four-form the `2`-rank is the
rank of the symmetric middle contraction/Pluecker quadric. Proposition 2.8
records the ranks `6, 10, 15` in six variables, via the dual classification of
two-forms. DOI: <https://doi.org/10.5802/aif.3131>.

This source supports the contraction-rank convention and a six-dimensional
control. It does not state the reconstructed `mu_4(m)` problem or a 16D
22/23 result.

### Cardinali--Giuzzi--Pasini (2017)

The paper treats alternating `k`-linear forms through hyperplanes of the
Pluecker-embedded Grassmannian and studies the subspaces on which contraction
vanishes. This supports the need to record the radical/support convention and
the base field. DOI: <https://doi.org/10.1007/s10801-016-0730-6>.

It does not identify the middle-rank extremum needed here.

### Arrondo--Bernardi--Macias Marques--Mourrain (2019)

This paper introduces skew apolarity for Grassmann/Skew decomposition rank,
with algorithms stated for degree at most three and ambient dimension at most
eight. DOI: <https://doi.org/10.1142/S0219199719500615>.

Its decomposition rank is not `rank C_2`; it is retained to prevent a silent
rank-notion substitution in later searches.

### Suciu (2020) and the exterior apolar quotient

Suciu defines a formal-dimension-`d` Poincare-duality algebra by perfect
multiplication pairings and associates an alternating top form to every such
graded skew-commutative algebra. The paper's bijection between alternating
forms and algebra is specifically a formal-dimension-three statement. It
cannot be quoted as a degree-four classification.

For the present problem one instead defines

```text
A_omega = Lambda(V) / Ann_wedge(omega),
Ann_wedge(omega)_j = ker C_j(omega).
```

The annihilator is a homogeneous ideal, and quotienting the left and right
kernels of the pairing `omega(a wedge b)` gives perfect complementary-degree
pairings. Thus `dim (A_omega)_j = rank C_j(omega)` and a concise four-form has
Hilbert vector `(1,m,r_2,m,1)`. A self-contained proof is now in
`math/four_forms/problem_statement.tex`; Suciu supplies the established PD
terminology, not the missing four-form extremum.

### Cohen--Helminck (1988): exact dimension seven

Theorem 2.1 and Table 1 classify the nine nonzero three-form orbits over an
algebraically closed field in dimension seven. Coordinate Hodge duality turns
these into four-form orbit representatives; projective orbit correspondence
uses the surjective contragredient automorphism of `GL(7)` and a determinant
scalar, not same-matrix equivariance. Exact rational
elimination gives middle ranks `6,10,12,12,16,15,18,15,21` and first ranks
`4,6,7,7,7,7,7,6,7`; hence the concise minimum is 12. The orbit `f_3=123+456`
has rational dual `4567-1237`, proving

```text
mu_4^Q(7) = mu_4^Qbar(7) = 12.
```

The classification source is responsible for exhaustiveness. The independent
artifact `seven_dimensional_orbit_ranks.json` and verifier are responsible for
the transcription, exterior signs, and exact rank calculations. Payload hash:
`94f1a654978dd1d37770b5a2171a07a5a839525dac1d16b6247a3b1ab2665f21`.

### Antonyan--Oeding and theta groups: exact dimension eight

Oeding's translation of Antonyan gives a seven-dimensional Cartan subspace for
`Lambda^4(C^8)` and Table 10 normal forms for all 94 nilpotent `SL(8,C)`
orbits. Galitski--Timashev's theta-group theorem supplies the orbit-closure
bridge: the semisimple part is the unique closed orbit in an orbit closure,
and every semisimple element is conjugate into a Cartan subspace.

For the seven recorded Cartan generators, the exact `C_2` matrices commute and
have an integral simultaneous eigenbasis with 28 distinct weights in
`{-1,0,1}^7`. An exhaustive check of all `3^7-1` nonzero normals over `F_3`
finds at most 16 weights in a hyperplane. This proves the complex bound, rather
than merely suggesting it: 17 integral weights in a complex hyperplane would
have integer row rank at most six, hence also rank at most six modulo three,
contradicting the finite enumeration. Thus every nonzero semisimple form has
middle rank at least 12.

The determinantal locus `rank C_2 < 12` is invariant and Zariski closed. If it
contained a form with nonzero semisimple part, its orbit closure would contain
a nonzero Cartan form in the same locus, contradicting the preceding bound.
Any remaining candidate is nilpotent. Exact reranking of all 94 source normal
forms finds 85 concise orbits and a minimum middle rank of 12; orbit 6 is the
unique concise nilpotent minimizer. Finally `1234+5678` is a rational concise
witness with ranks `(1,8,12,8,1)`. Therefore

```text
mu_4^C(8) = mu_4^Qbar(8) = mu_4^Q(8) = 12.
```

The independent artifact
`math/four_forms/eight_dimensional_four_form_minimum.json` records all 94 rows,
the Cartan joint eigenbasis, finite-field enumeration, and witness. Its
mathematical-payload hash is
`44288f6097c7f56c746f3e3c39885fe707704acf47b957129e786afab044214b`;
its source-transcription hash is
`bde922dcdf7766082b1fc2bb8d7f844ae24dff7aa0fe381504cb5cc68a453648`.
The source theorems, not the verifier, supply orbit coverage and Jordan/closure
theory.

### A rejected provenance lead: commutative Gorenstein vectors

Migliore--Zanello classify ordinary commutative Artinian Gorenstein Hilbert
vectors `(1,r,h_2,r,1)` through `r<=17`; at `r=16` their minimum is 15. This is
the symmetric-polynomial/Macaulay setting, not a quotient of an exterior
algebra by an alternating four-form. It neither supplies the values 22/23 nor
provides a valid lower bound for the present problem.

## Exact elementary controls

- A coordinate volume form in dimension four has ranks `(1,4,6,4,1)`.
- A nonzero four-form in dimension five has support at most four, so no concise
  five-dimensional example exists.
- In dimension six, the Hodge dual of a nondegenerate two-form is concise and
  has ranks `(1,6,15,6,1)`, agreeing with the published six-variable rank
  classification.
- In dimension seven, source-complete orbit coverage and an independent exact
  rank-table certificate prove `mu_4^Q(7)=mu_4^Qbar(7)=12`.
- In dimension eight, the Cartan/closure reduction and all 94 nilpotent normal
  forms prove `mu_4^C(8)=mu_4^Qbar(8)=mu_4^Q(8)=12`.
- Direct sums of `t` disjoint four-dimensional volume forms have ranks
  `(1,4t,6t,4t,1)`. In particular dimension 16 has a rational rank-24 control.

These are **theorem**-level linear-algebra controls, not a sharp 16D result.
The exact implementation and tests live in `math/four_forms/` and
`tests/test_four_form_contractions.py`.

## Open provenance checklist

- Continue searching for prior exterior/skew inverse-system terminology for
  the now self-contained perfect-pairing proposition; do not make priority
  claims from absence of a source.
- Find sharp rank-stratum results for four-forms in dimensions nine and above.
- Audit orbit classifications separately over `Q`, `R`, `C`, and finite fields.
- Locate the origin of the 16D alternatives 22 and 23; absent that, construct a
  candidate from scratch and label the search as exploratory.
- Do not infer a characteristic-zero upper bound from a finite-field rank drop.
