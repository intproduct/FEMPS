# Active execution plan: Phase 25 statistics-carrier/multiplicity gate

## Objective

Test the master-plan hypothesis that exchange statistics can be isolated in a
fixed structural carrier while a smaller, gauge-independent multiplicity
space stores only correlations beyond a Slater determinant. Determine whether
this produces a mathematically invariant and algorithmically useful structure,
or merely renames the ordinary particle Schmidt/exterior contraction cost.

## Candidate L1

At every particle cut `k|N-k`, seek a functorial factorization or quotient of
the exterior contraction map of the form

```text
B_k  ~=  S_k^fermion tensor C_k^corr,
```

with `dim C_k^corr=1` for every nonzero Slater determinant, invariance under
one-particle basis changes and internal FEMPS gauges, and a reconstruction/
contraction rule that never materializes the binomial statistics carrier.

## Checkpoints

- [ ] Define precisely which object is factored: particle-cut Schmidt support,
  image of exterior contraction, Pluecker/Grassmannian tangent data, or a
  representation-category multiplicity space.
- [ ] Audit symmetry-adapted TN structural/degeneracy decompositions, Schur--
  Weyl/exterior representation theory, fermionic Gaussian/Pfaffian canonical
  forms, and Grassmannian secant/tangent constructions.
- [ ] Prove the single-Slater sanity condition without choosing occupied
  orbitals or a noncanonical gauge as hidden input.
- [ ] Determine behavior under finite Slater sums and generic APG/matrix-pair
  states; compare the proposed multiplicity with Slater/secant rank and the
  existing contribution-Gram diagnostic.
- [ ] Test uniqueness, functoriality, gauge invariance, direct sums, orbital
  rotations, and stability under small perturbations on exact small systems.
- [ ] Derive reconstruction, norm, one-body, and factorized-two-body costs in
  `(N,D,chi_corr)` while accounting explicitly for the structural carrier.
- [ ] Check whether the Phase 13/22 permanent embeddings survive unchanged in
  the proposed quotient/factorization.
- [ ] Require a safe truncation theorem: antisymmetry remains exact and a
  discarded multiplicity weight bounds state/observable error.
- [ ] Issue Gate L before any new optimizer or GPU benchmark.

## Exit criterion

Gate L passes only if the correlation multiplicity is canonical up to unitary
equivalence, gives Slater multiplicity one, admits reconstruction and
observable contraction without hidden binomial/exponential state, and supports
a symmetry-preserving truncation error bound. A proof that no such factorization
can be both functorial and contractible in the proposed category is an
acceptable negative outcome.
