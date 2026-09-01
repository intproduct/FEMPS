# CHSS reduction audit for the fixed-bond FEMPS theorem

## Status

This is a theorem-by-theorem source audit of Chien--Harsha--Sinclair--
Srinivasan (CHSS), not a replacement proof of their gadget lemmas. The source
checked is *Almost Settling the Hardness of Noncommutative Determinant*,
Theorems 3.5 and 3.9. External human algebraic-complexity review is pending.

## Exact source statement used

For a 3SAT formula `phi` with `m` clauses and `S` satisfying assignments,
CHSS construct a polynomial-size directed graph `H_phi` with edge weights in
the fixed algebra `Mat_2(F)`. With vertices numbered in the Cayley row order,

```text
CDet(H_phi) = a I_2 + b J_2,
J_2 = [[0,1],[1,0]],
a+b = 4^(3m) S.
```

CHSS Theorem 3.5 states #P-hardness in characteristic zero and
`Mod_p P`-hardness in positive characteristic `p>2`. The FEMPS theorem
specializes to `F=Q`; characteristic two is excluded.

The construction uses a polynomial number of constant-size gadgets. Their
displayed weights are constant-bit integer `2 x 2` matrices, so the adjacency
matrix and every resulting FEMPS core have polynomial rational encoding length.

## Boundary and norm recovery

Set `u=e_1` and `v=e_1+e_2`. Then

```text
u^T (a I_2+b J_2) v = a+b = 4^(3m) S >= 0.
```

The direct Cayley/exterior coefficient identity produces a one-form FEMPS with
`D=N`, maximum internal bond two, and the sole top-form coefficient
`4^(3m)S`. Its squared norm is therefore `16^(3m)S^2`. One exact norm query,
an exact nonnegative integer square root, and division by `4^(3m)` recover
`S`. Since `S<=2^nvar`, the oracle output, square root, and postprocessing all
have polynomial bit length. This is a polynomial-time metric reduction (and
hence also a Turing reduction).

## What the older bond-three argument still proves

For an arbitrary signed Cayley entry `x`, direct-summing the bond-two state
with a bond-one unit top form gives maximum bond three and coefficient `x+1`.
The two norms `x^2` and `(x+1)^2` recover
`x=((x+1)^2-x^2-1)/2`. This remains a valid general-output polarization
argument, but it is not the sharp CHSS hardness boundary.

## Encoding checklist

- base field in the manuscript theorem: `Q`;
- source problem: exact `#3SAT`;
- target algebra: fixed `Mat_2(Q)`, vector-space dimension four;
- virtual maximum bond: two after endpoint-boundary absorption;
- row order: CHSS graph vertex order equals Cayley multiplication order;
- source-to-target size: polynomial number of constant-size gadgets;
- coefficient size: constant-bit gadget entries, hence polynomial total input;
- oracle queries: one exact squared norm;
- postprocessing: exact integer square root and power-of-four division;
- output bit length: polynomial because `S<=2^nvar`;
- excluded inference: no conclusion about approximate contraction, QMC, or
  every promised `chi>=2` family.

## Evidence boundary

The existing repository verifier checks the direct Cayley coefficient identity,
four generic boundaries, bond-three polarization, and antisymmetry for small
orders. It does **not** instantiate or independently certify the CHSS gadgets.
The fixed-bond theorem therefore depends on the published CHSS result and still
requires external expert review before submission.

