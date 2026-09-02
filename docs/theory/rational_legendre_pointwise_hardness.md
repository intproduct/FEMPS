# Exact pointwise hardness in a rational shifted-Legendre basis

## Scientific status

The reduction is now stated and proved as a **theorem** in the canonical Paper
A source. This promotion closes the final repository-internal proof pass; it
does not constitute the still-required independent human algebraic-complexity
review. It is not a statement about approximate sampling, VMC, QMC, or every
`chi>=2` FEMPS instance.

## Exact evaluation problem

For `ell_r(t)=P_r(2t-1)`, where `P_r(1)=1`, the input consists of:

- rational coefficient matrices for site-labelled one-form FEMPS cores in
  `ell_0,...,ell_(D-1)`;
- rational boundaries and rational query points;
- maximum virtual bond two and, in the hard family, `D=N`.

With the isometric exterior map, every such rational query value has the form
`q/sqrt(N!)`. The exact output is the pair `(q,N)`, with rational `q`;
equivalently, the problem may request the unnormalized rational alternating
value `q`. This explicitly represents the only algebraic normalization factor
and invokes no hidden algebraic-number oracle.

## Evaluation matrix and bit bounds

The integer expansion

```text
ell_r(t) = sum_(k=0)^r (-1)^(r-k) binom(r,k) binom(r+k,k) t^k
```

has leading coefficient `binom(2r,r)`. For `t_i=i/(N+1)` and
`B_ij=ell_(j-1)(t_i)`, degree grading gives

```text
det(B) = product_(r=0)^(N-1) binom(2r,r)
         product_(i<k) (t_k-t_i) != 0.
```

The nodes have `O(log N)` bits. The integer coefficient formula makes every
evaluation polynomial-bit. The common denominator `Q=(N+1)^(N-1)` clears all
entries. Hadamard bounds for `QB` and its minors then show that `det(B)`, its
reciprocal, and

```text
B^(-1) = Q adj(QB) / det(QB)
```

have polynomial bit length and are computable by exact rational arithmetic in
polynomial time. The inverse is not needed by the direct basis reduction, but
the bound also certifies that passage to point-value/Lagrange coordinates has
no hidden exponential encoding cost.

## Independent reduction from CHSS

For the CHSS matrix `H=(H_ij)` over `Mat_2(Q)`, define

```text
A^[i](t) = sum_(j=1)^N H_ij ell_(j-1)(t),
u=e_1, v=e_1+e_2.
```

The CHSS entries themselves are the functional-basis coefficients. Expanding
at `(t_1,...,t_N)`, scalar evaluation factors commute while the virtual
matrices remain row ordered, so

```text
Psi(t_1,...,t_N)
 = det(B) u^T CDet(H) v / sqrt(N!)
 = det(B) 4^(3m) #SAT(phi) / sqrt(N!).
```

Repeated basis-column choices cancel; a column permutation contributes its
sign times `det(B)`. All inputs, the query, the exact output representation,
and the postprocessing have polynomial bit length. One exact query followed by
rational division therefore recovers `#SAT`, giving a polynomial-time metric
reduction.

This pointwise theorem and the fixed-bond squared-norm theorem are independent
reductions from the same Cayley source. Neither is deduced from the other.
The bounded verifier checks the determinant, inverse, and point formula for
orders two through six; that **exact certificate** guards against
transcription errors but does not replace the arbitrary-size proof.
