# Gate A contraction literature boundary

## Fermionic signs are not a contraction shortcut

Barthel--Pineda--Eisert formulate fermionic operator circuits whose contraction
has the same leading computation and memory requirements as the corresponding
qudit circuit, with marginal overhead for mode reorderings. Their analysis also
states that exact intermediate boundary states can grow exponentially in the
same geometries where ordinary tensor-network contraction does. This supports
using Grassmann/graded machinery for correctness without treating it as an
automatic complexity reduction.

## Gaussian/matchgate closure is the genuine polynomial mechanism

Gaussian fermionic states and matchgate tensors close under contraction at the
level of covariance or antisymmetric generating matrices. Jahn et al. use this
closure for efficient matchgate networks; Schuch--Bauer develop MPS algorithms
directly in the Gaussian formalism. The key resource is Wick/Gaussian closure,
not fermionic parity alone.

A pure number-conserving Gaussian state at fixed particle number is a single
Slater determinant, matching the `chi=1` FEMPS limit. Number projection of a
pairing Gaussian produces Pfaffian/antisymmetrized-geminal structure and is a
legitimate nontrivial fallback candidate, but it is a restricted ansatz rather
than evidence that generic matrix-wedge FEMPS contracts polynomially.

## Consequence for the project

The next contraction search should test Gaussian/Pfaffian closure or a bounded
non-Gaussian extension explicitly. Re-expressing the current generic cores in
Grassmann variables without such closure would only reproduce the exterior
Fock-space dimension already measured by the dynamic program.

## Noncommutative determinant boundary

Chien--Harsha--Sinclair--Srinivasan prove permanent hardness for the
row-ordered determinant already over `2 x 2` matrix entries. Their finite-
algebra easy side assumes a commutative quotient by the radical and has runtime
`N^O(d)`, where `d` is the radical nilpotency index. The main classification is
over finite fields; their appendix records the bounded-nilpotency extension for
rational input algebras. Consequently, an upper-triangular bond size that grows
with particle number is not a jointly polynomial consequence of that theorem.

Phase 13 supplies the previously missing project-specific link: growing shift
tags embed the hard row-ordered determinant into one symmetrized matrix-pair
amplitude and then into a generic FEMPS norm. Phase 20 checks the opposite
boundary. For `T2`, whose radical squares to zero, the entire arbitrary-boundary
matrix-pair state collapses over characteristic zero to `O(M^2)` scalar AGPs.
Thus the smallest noncommuting determinant-easy algebra is also tractable here,
but only through an explicit polynomial-size LC-AGP organization. The same
homogeneous-power argument gives `O(M^3)` AGPs for the fully noncommutative
semisimple algebra `Mat_2`. A Wedderburn--radical word expansion then proves the
general pair-state boundary: fixed maximum simple-block size and fixed radical
nilpotency index imply a polynomial-size LC-AGP expansion, even when the
semisimple quotient is noncommutative. This project result is independent of,
and should not be attributed to, the row-ordered determinant theorem.
