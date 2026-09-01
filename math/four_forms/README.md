# Exact four-form research workspace

This directory is the proof-oriented workspace for the Phase 27 alternating
four-form rank-spectrum program. It is deliberately independent of the
production `femps` package and of PyTorch.

## Evidence status

- **Theorem / source-backed convention:** for a four-form
  `omega in Lambda^4(V*)`, the `j`-th contraction map is
  `C_j(omega): Lambda^j(V) -> Lambda^(4-j)(V*)`. The `j`-rank is its matrix
  rank. De Poi--Faenzi--Mezzetti--Ranestad use this convention and identify the
  `2`-rank with the rank of the corresponding Pluecker quadric. See Definition
  2.4 and the discussion preceding it in
  <https://doi.org/10.5802/aif.3131>.
- **Theorem / elementary linear algebra:** `C_2` is symmetric because two
  2-vectors commute in the exterior algebra, while `C_3 = -C_1^T` in the basis
  convention implemented here. Consequently the contraction-rank vector of a
  concise four-form is `(1, m, r_2, m, 1)`.
- **Theorem / exterior apolarity:** with
  `A_omega = Lambda(V)/Ann_wedge(omega)` and
  `Ann_wedge(omega)_j = ker C_j(omega)`, multiplication gives perfect
  complementary-degree pairings and
  `dim (A_omega)_j = rank C_j(omega)`. The self-contained statement and proof
  are in `problem_statement.tex`.
- **Theorem / source-backed orbit coverage plus exact certificate:**
  Cohen--Helminck's nine seven-dimensional trivector orbits, transported by a
  coordinate volume dual and reranked exactly, give
  `mu_4^Q(7) = mu_4^Qbar(7) = 12`.
- **Theorem / source-backed orbit and closure coverage plus exact certificate:**
  Antonyan--Oeding's eight-dimensional Cartan subspace and 94 nilpotent
  normal forms, together with the theta-group orbit-closure theorem, reduce
  every possible low-rank form to two exact checks. A simultaneous Cartan
  eigenbasis and an `F_3` hyperplane certificate bound every nonzero
  semisimple form by rank 12; exact reranking of all nilpotent normal forms
  gives the same concise lower bound. Hence
  `mu_4^C(8) = mu_4^Qbar(8) = mu_4^Q(8) = 12`.
- **Working reconstruction, not an inherited theorem:** until a primary source
  or old certificate is recovered, this project writes

  ```text
  mu_4^K(m) = min rank_K C_2(omega)
              over omega in Lambda^4(K^m)* with rank_K C_1(omega) = m.
  ```

  This definition makes the master plan's “16D rank 22/23” branch the question
  whether `mu_4^K(16)` is `22`, `23`, or obeys a different certified bound. The
  notation and the numerical alternatives are currently a **conjectural target**:
  no candidate, proof, script, or certificate was found in the repository or
  its Git history.

## Conventions

- Basis indices are zero-based integers.
- A form is a dictionary from increasing tuples to integer or rational
  coefficients.
- `contraction_matrix(omega, m, j)` has columns indexed by increasing
  `j`-subsets `I`, rows indexed by increasing `(4-j)`-subsets `J`, and entry
  `omega(e_I wedge e_J)`.
- Rational rank means exact Gaussian elimination over `Q`; finite-field rank
  records the prime explicitly.
- The quadratic interpretation of `C_2` is restricted to characteristic not
  two. Integer/rational calculations are the default. A finite-field rank is
  evidence only over the recorded field unless a separate lifting argument is
  supplied.
- “Contraction rank” is not Grassmann/Slater rank, tensor rank, border rank, or
  an ordinary TT rank. Those notions must never be interchanged silently.
- “Concise” means `rank C_1 = m`, equivalently zero contraction radical in the
  chosen `m`-dimensional ambient space.
- Two forms are `GL(V,K)`-equivalent when one is obtained from the other by the
  natural change-of-basis action of some `g in GL(V,K)`. Projective orbit
  equivalence additionally allows multiplication by a nonzero scalar. All
  rank and conciseness statements are invariant under either relation, but a
  certificate must state which orbit relation and field it uses.

## Reproduction

From the repository root:

```powershell
python -m pytest -q tests/test_four_form_contractions.py
```

`exact_contractions.py` uses only the Python standard library. Single-form
certificate artifacts validate against `certificate.schema.json`; a
classification table may use a purpose-specific schema enforced by its
independent verifier. Every proof artifact records its field, hashes its
mathematical payload, and is checked by a verifier that does not import the
exact utility module.

The first such artifact is `direct_sum_16_rank24_certificate.json`. Verify it
independently with:

```powershell
python math/four_forms/verify_direct_sum_control.py --verify math/four_forms/direct_sum_16_rank24_certificate.json
```

Its mathematical-payload SHA-256 is
`3e48d8e9e0ed1802805d5446c573cef7daca05146abae45d90679fa5a633edcd`.

The seven-dimensional source-orbit table is independently checked with:

```powershell
python math/four_forms/verify_seven_dimensional_classification.py --verify math/four_forms/seven_dimensional_orbit_ranks.json
```

Its payload SHA-256 is
`94f1a654978dd1d37770b5a2171a07a5a839525dac1d16b6247a3b1ab2665f21`.
The script verifies the source transcription and exact ranks; the cited
Cohen--Helminck theorem, rather than the script, supplies orbit exhaustiveness.

The eight-dimensional semisimple/nilpotent certificate is independently
checked with:

```powershell
python math/four_forms/verify_eight_dimensional_minimum.py --verify math/four_forms/eight_dimensional_four_form_minimum.json
```

Its mathematical-payload SHA-256 is
`44288f6097c7f56c746f3e3c39885fe707704acf47b957129e786afab044214b`;
the 94-row source-transcription SHA-256 is
`bde922dcdf7766082b1fc2bb8d7f844ae24dff7aa0fe381504cb5cc68a453648`.
The verifier recomputes all contraction ranks, Cartan joint eigenpairs, and
the finite-field hyperplane enumeration. Antonyan--Oeding Table 10 and the
theta-group theorem remain the separate orbit-coverage inputs.

## Current boundary

The controls now establish the exact seven- and eight-dimensional values in
addition to the earlier low-dimensional consistency checks. They do not
establish a sharp value of `mu_4(16)`. In particular, the direct
sum of four disjoint volume forms is a concise rational 16-dimensional control
of middle rank 24, hence only the elementary upper bound
`mu_4^Q(16) <= 24`. It is not evidence for rank 22 or 23.
