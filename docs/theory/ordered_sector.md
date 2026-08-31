# Ordered-coordinate sector as a competing first-quantized representation

## Evidence level and scope

This document fixes definitions for the Phase 8 comparator. The Hilbert-space
equivalence below is an elementary exact statement. Claims about MPS ranks,
optimization cost, or superiority of one representation remain numerical and
must be measured separately.

## Normalized Weyl-chamber map

Let

\[
W=\{(x_1,\ldots,x_N)\in\mathbb R^N:x_1<\cdots<x_N\}.
\]

For a normalized antisymmetric wavefunction `Psi`, define

\[
(U\Psi)(x_1,\ldots,x_N)=\sqrt{N!}\,\Psi(x_1,\ldots,x_N),
\qquad x\in W.
\]

The coincidence hyperplanes have measure zero and the `N!` permutation
chambers have equal norm, hence `U` is an isometry from the antisymmetric
full-space Hilbert space to `L^2(W)`. Its inverse sorts a configuration into
`W`, multiplies by the permutation sign, and divides by `sqrt(N!)`.

For regular fermionic wavefunctions the ordered representative has Dirichlet
boundary values at `x_i=x_(i+1)`. This hard wall, not an internal sign tensor,
carries Pauli exclusion. A permutation-invariant local Schrödinger operator is
unitarily equivalent to its restriction to `W` with that boundary condition.

## Interparticle-distance coordinates

Set

\[
q_1=x_1,\qquad q_i=x_i-x_{i-1}>0\quad(i>1),
\]

so `x_i=sum_(a<=i) q_a` and the Jacobian is one. On the full line,
`q_1` is real and there is no box-sum constraint. For the finite lattice box in
Li--Waintal 2026, all `q_i` are positive integers and `sum_i q_i <= L`.

The continuous derivatives transform as

\[
\partial_{x_i}=\partial_{q_i}-\partial_{q_{i+1}}\ (i<N),
\qquad \partial_{x_N}=\partial_{q_N},
\]

and therefore

\[
\sum_i\partial_{x_i}^2=
\partial_{q_1}^2+2\sum_{i=2}^N\partial_{q_i}^2
-2\sum_{i=1}^{N-1}\partial_{q_i}\partial_{q_{i+1}}.
\]

The kinetic operator is nearest-neighbor in the distance variables. A harmonic
trap becomes a dense quadratic form in cumulative sums of `q`; a pair potential
depends on interval sums `x_j-x_i=sum_(a=i+1)^j q_a`. Thus the ordered-sector
route removes exchange multiplicity but does not automatically guarantee a
small MPO for every continuum potential.

## Exact discrete oracle

`src/femps/ordered_sector.py` supplies two deliberately small truth paths:

1. normalized restriction and antisymmetric extension of an explicit particle
   tensor;
2. direct hard-wall projection of a tridiagonal coordinate-grid Hamiltonian
   onto strictly increasing grid configurations.

For local hopping, a particle cannot jump over another particle without first
touching the excluded collision boundary. The ordered matrix is then exactly
the same matrix as the independent exterior Slater--Condon lift in the
increasing coordinate basis. Dense/nonlocal one-body matrices are rejected:
projecting those without permutation signs would not discretize the local
Weyl-chamber Schrödinger operator.

## Comparison protocol

| Representation | Stored domain | Antisymmetry mechanism | Required controls |
|---|---|---|---|
| Ordinary particle TT | all labeled coordinates | explicit sign representation in coefficients | `D`, every particle bond, symmetry residual |
| Ordered-sector FTN | one Weyl chamber | hard collision boundary plus signed extension | coordinate/basis resolution, boundary leakage, MPS bonds |
| Pfaffian/finite-AGP FEMPS | full functional one-particle space | exterior/Pfaffian structural carrier | `D`, pair rank or AGP length `K`, overlap conditioning |

Energy comparisons must use the same physical Hamiltonian and report continuum,
basis/discretization, ansatz, and optimizer errors separately. Parameter count
or a single bond label is not by itself a complexity comparison.

## Relation to prior work

Li and Waintal, *Phys. Rev. Lett.* **136**, 116503 (2026), introduce the
ordered lattice sector and positive interparticle distances, then enforce the
finite-box sum constraint with a projector MPO, a monitored Lagrange penalty,
and a controlled distance cutoff. FEMPS must not claim priority for efficient
first-quantized fermionic MPS. Its narrower distinction is the 2201 orthonormal
functional-basis calculus combined with a full-space exterior carrier.
