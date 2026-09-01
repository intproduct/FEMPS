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

## Exact elementary controls

- A coordinate volume form in dimension four has ranks `(1,4,6,4,1)`.
- A nonzero four-form in dimension five has support at most four, so no concise
  five-dimensional example exists.
- In dimension six, the Hodge dual of a nondegenerate two-form is concise and
  has ranks `(1,6,15,6,1)`, agreeing with the published six-variable rank
  classification.
- Direct sums of `t` disjoint four-dimensional volume forms have ranks
  `(1,4t,6t,4t,1)`. In particular dimension 16 has a rational rank-24 control.

These are **theorem**-level linear-algebra controls, not a sharp 16D result.
The exact implementation and tests live in `math/four_forms/` and
`tests/test_four_form_contractions.py`.

## Open provenance checklist

- Find an exterior/skew Artinian Gorenstein source that states the perfect
  pairing/Hilbert vector for a principal alternating inverse system.
- Find sharp rank-stratum results for four-forms beyond the six-variable case.
- Audit orbit classifications separately over `Q`, `R`, `C`, and finite fields.
- Locate the origin of the 16D alternatives 22 and 23; absent that, construct a
  candidate from scratch and label the search as exploratory.
- Do not infer a characteristic-zero upper bound from a finite-field rank drop.
