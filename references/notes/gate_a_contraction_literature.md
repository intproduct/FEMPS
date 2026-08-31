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
