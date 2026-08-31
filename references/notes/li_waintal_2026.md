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

## Phase 16 implementation comparison

The shared structural step is exact: keep one ordering chamber and let positive
interparticle distances impose Pauli exclusion. The implementations then
diverge in several auditable ways:

- Li--Waintal use `q1=x1` and `qn=xn-x(n-1)` on a coordinate grid. Phase 16
  uses the center of mass plus `N-1` gaps. Its Jacobian has magnitude one, the
  center of mass decouples from the kinetic metric, and a confining trap needs
  no cumulative finite-box projector.
- Their local physical index is a truncated integer distance, controlled by
  `Qmax` (and sometimes `Qmin`). Phase 16 uses orthonormal continuous
  functions: a Dirichlet sine box with independent `D,Rmax` controls and an
  unbounded odd-Hermite candidate with independent `D,length_scale` controls.
- Their hopping becomes adjacent distance shifts and their finite-range
  density interaction has MPO rank `r+1`. Here the continuum Laplacian becomes
  a Cartan mixed-derivative MPO, and every soft-Coulomb pair is approximated by
  a degree-`K` interval-polynomial automaton; the direct all-pair bond is
  conservatively `O(N^2 K)`.
- Their reported ground-state and dynamics engines are DMRG and TDVP. The
  present prototype deliberately retains Hong et al.'s global automatic
  differentiation and uses latticeTN only through native MPS/MPO contractions.

These are integration and error-control differences, not grounds for claiming
the ordered chamber or distance MPS as new. The current soft-Coulomb automaton
also applies only to the finite sine interval; extending the unbounded basis to
interactions remains a next-phase task.

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

Gate A has since conditionally obstructed generic matrix-wedge contraction.
The surviving ordered-distance result must therefore be named and evaluated as
an efficient competing first-quantized representation, not relabeled FEMPS.
