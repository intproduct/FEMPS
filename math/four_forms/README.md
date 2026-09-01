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

`exact_contractions.py` uses only the Python standard library. Certificate
artifacts added later must validate against `certificate.schema.json`, include
their base field and characteristic, hash their mathematical payload, and be
checked by a second verifier that does not import this module.

The first such artifact is `direct_sum_16_rank24_certificate.json`. Verify it
independently with:

```powershell
python math/four_forms/verify_direct_sum_control.py --verify math/four_forms/direct_sum_16_rank24_certificate.json
```

Its mathematical-payload SHA-256 is
`3e48d8e9e0ed1802805d5446c573cef7daca05146abae45d90679fa5a633edcd`.

## Current boundary

The committed controls establish definitions and low-dimensional consistency;
they do not establish a sharp value of `mu_4(16)`. In particular, the direct
sum of four disjoint volume forms is a concise rational 16-dimensional control
of middle rank 24, hence only the elementary upper bound
`mu_4^Q(16) <= 24`. It is not evidence for rank 22 or 23.
