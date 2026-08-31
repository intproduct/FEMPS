# Li and Waintal 2026 - Matrix Product States and First Quantization

## Representation

The method keeps only the ordered coordinate sector
`0 < x1 < ... < xN < L+1`, then changes variables to interparticle distances
`q1=x1`, `qn=xn-x(n-1)`. Positivity of `q` implements Pauli exclusion, while
the wavefunction in other particle-ordering sectors can be recovered by
permutation. The MPS sites are particles/distances rather than orbitals.

The box constraint is enforced by a projector MPO with controlled
approximations: a Lagrange penalty and a local distance cutoff `Qmax`. Kinetic
terms couple adjacent `q` variables and admit a small MPO; finite-range
interactions have MPO rank `r+1`.

## Relation to FEMPS

This is the clearest implementation of the Master Plan's ordered-sector or
Weyl-chamber alternative. It removes the exchange multiplicity by representing
only independent coordinate orderings, rather than storing an explicit
antisymmetric tensor or an exterior structural carrier. It is first-quantized
and efficient, so FEMPS must not claim broadly that it is the first efficient
first-quantized MPS for fermions.

The remaining possible FEMPS distinction is narrower: retain the 2201
orthonormal continuous functional-basis operator calculus and a full-space
state that is antisymmetric by exterior construction, with a separate
correlation multiplicity. Whether that distinction is useful depends on Gate A.

