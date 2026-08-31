# Virtual-matrix-valued pair powers

## Status and evidence level

This note defines the smallest Phase 13 candidate beyond an explicit list of
scalar AGPs. Algebraic identities are theorem statements backed by exact
PyTorch tests. Polynomial contractibility and a lower bound on LC-AGP length
are **not** established.

## Definition

Let `B_ij` for `i<j` be `chi x chi` matrices, extended by

`B_ji = -B_ij`, `B_ii = 0`.

Introduce the virtual-matrix-valued two-form

`Omega_B = sum_{i<j} B_ij e_i wedge e_j`.

For boundaries `l,r in C^chi`, define the `N=2M` exterior state

`Psi_M(B;l,r) = l^T (Omega_B^M / M!) r`.

Matrix multiplication is retained in the displayed factor order. Physical
two-forms have even degree and commute, so repeated factors symmetrize the
matrix products. The ansatz has `O(D^2 chi^2)` displayed parameters before
skew symmetry and gauge removal, independent of M.

The exact support recurrence implemented by
`matrix_pair_exterior_matrices` is

`C_empty^(0) = I`,

`C_S^(m) = (1/m) sum_{T subset S, |T|=2} sign(S\T,T) C_(S\T)^(m-1) B_T`.

The physical increasing-basis coefficient is `c_S=l^T C_S r`.

## N=4 noncommutative Pfaffian

For `i<j<k<l`, direct expansion gives

`Q_ijkl = 1/2 ( {B_ij,B_kl} - {B_ik,B_jl} + {B_il,B_jk} )`,

where `{X,Y}=XY+YX`, and `c_ijkl=l^T Q_ijkl r`. This is the
fully symmetrized noncommutative analogue selected by the exterior power; it
is not assumed to satisfy any quantum-group or Manin-matrix relations.

The exact N=4 norm is

`<Psi|Psi> = sum_{i<j<k<l} |l^T Q_ijkl r|^2`.

For a one-body matrix `h`, the second-quantized expression is

`<Psi|dGamma(h)|Psi> = sum_(S,T) conjugate(c_S) H^(1)_(S,T) c_T`,

where `H^(1)_(S,T)` is zero unless S and T differ by at most one occupied
orbital, and otherwise equals the corresponding `h_pq` times the standard
annihilation/insertion parity. The executable check constructs this expression
in the increasing exterior basis and independently compares it with
`sum_a h(a)` acting on the full antisymmetric `D^4` particle tensor.

## Exact reductions and gauge

- `chi=1` is exactly scalar fixed-number AGP.
- If all `B_ij` are simultaneously diagonalizable,
  `B_ij=S diag(f_ij^1,...,f_ij^chi) S^-1`, then the state is an LC-AGP of at
  most chi terms, with weights `(l^T S)_a (S^-1 r)_a`.
- Simultaneous triangularizability alone is weaker: nilpotent off-diagonal
  pieces can produce derivative-like mixed contributions, so the diagonal
  LC-AGP statement must not be asserted without diagonalizability.
- The similarity action `B_ij -> G^-1 B_ij G`, `l^T -> l^T G`, and
  `r -> G^-1 r` leaves the state invariant.

The first possible non-scalar case is therefore `N=4,chi=2` with genuinely
noncommuting coefficient matrices. Noncommutativity is necessary to escape the
simultaneously diagonalizable reduction, but is not sufficient to prove a
large minimal LC-AGP rank.

## Current exact cost and contraction obstruction

The support recurrence has time

`O(chi^3 sum_(m=1)^M m^2 binom(D,2m))`

and peak memory

`O(chi^2 max_(m<=M) binom(D,2m))`.

It is polynomial for fixed N but combinatorial when N and D grow together.
Thus it is only a truth oracle. The scalar/determinantal closure at `chi=1`
and the simultaneously diagonalizable LC-AGP closure are known polynomial
subclasses. For generic noncommuting matrices, ordinary Pfaffian/determinant
identities cannot simply move coefficient factors past each other.

This observation is not a no-go theorem. Noncommutative Pfaffians and
quasideterminants exist under several algebraic definitions, often with
special commutation or quantum-group relations. Phase 13 must determine
whether the particular fully symmetrized finite-matrix construction above has
a double-layer closure polynomial jointly in `(N,D,chi)`. Until then it is a
candidate and an obstruction probe, not a production FEMPS solver.

## Tagged reduction and generic contraction obstruction

The connection to the row-ordered noncommutative determinant can be made with
polynomial-size shift tags. Let `A_ij` be entries of an `n x n` matrix over
`Mat_d`. Enlarge the virtual space to `C^(n+1) tensor C^d` and define

`B_(x_i,y_j) = E_(i,i+1) tensor A_ij`,

with all same-side pair coefficients zero. In the symmetrized product of n
selected cross edges, every factor order vanishes except row order
`1,2,...,n`, because only

`E_12 E_23 ... E_(n,n+1) = E_(1,n+1)`

is nonzero. With tag endpoints selected by the boundaries, the unique
top-degree coefficient at `D=N=2n` is

`(-1)^(n(n-1)/2) / n! * l^T CayleyDet(A) r`,

where

`CayleyDet(A) = sum_sigma sign(sigma) A_(1,sigma(1)) ... A_(n,sigma(n))`.

The construction uses pair virtual size `(n+1)d`. Chien--Harsha--Sinclair--
Srinivasan prove that evaluating this row-ordered determinant is as hard as
the permanent already for fixed `d=2`. Boundary choices recover its four
matrix entries with a constant number of queries.

This also obstructs a generic exact norm oracle, not only amplitude
evaluation. At top degree the physical exterior space is one-dimensional. A
virtual direct sum adds a known scalar reference amplitude to the hard
amplitude `x`; over real exact data, norms of `x` and `x+1` recover

`x = (|x+1|^2 - |x|^2 - 1) / 2`.

Finally, every matrix-pair power embeds into the original one-form
matrix-wedge FEMPS by using two particle cores per pair factor. The bonds
alternate between `chi` and `chi*D`; the first and last boundaries remain one.
For the tagged reduction, `D=N=2n`, pair `chi=2(n+1)`, and the largest original
FEMPS bond is `O(n^2)`. Therefore a generic exact norm contraction polynomial
jointly in `(N,D,chi_FEMPS)` would give a polynomial algorithm for the
permanent.

This is a conditional algebraic-complexity obstruction, not an unconditional
separation of complexity classes. It applies to the unrestricted dense
noncommuting family. It does not rule out physically meaningful restricted
coefficient algebras, approximation schemes, or ordered-sector formulations.

## Gauge-invariant correlation multiplicity

For any normalized fixed-N state let `gamma` be its one-body density matrix,
with `Tr(gamma)=N`, and define

`mu_1 = N / Tr(gamma^2)`.

Natural occupations lie in `[0,1]`, so `mu_1 >= 1`; equality holds exactly
when the one-body density is an N-dimensional projector, in particular for a
single Slater determinant. The quantity is invariant under virtual gauges and
one-particle basis rotations. We call it a one-body correlation multiplicity,
not entanglement: it does not determine higher-order correlation and is not a
newly invented invariant. The implementation operates on exact exterior
coefficients and is currently a diagnostic oracle.

## Executable evidence

`tests/test_matrix_pair.py` verifies:

1. `chi=1` equality with scalar AGP;
2. recurrence equality with the N=4 anticommutator formula;
3. diagonal virtual matrices equal an explicit chi-term LC-AGP;
4. similarity-gauge invariance;
5. finite reverse-mode gradients;
6. N=4 norm and one-body agreement between exterior and full particle-tensor
   routes;
7. the tagged Cayley-determinant identity for `n=1,2,3`; and
8. the polynomial-bond embedding into one-form matrix-wedge FEMPS.
