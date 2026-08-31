# Li and Chan 2016 - Hilbert-space MPS

## Representation

HS-MPS rotates the configuration graph so renormalization proceeds by electron
number rather than orbital number. Its MPS has `N` electron sites and orbital
physical indices `p_i`, but only the strictly ordered independent coefficients
`p1 < ... < pN` are stored. Prefix/suffix constraints identify the orbital
support of renormalized states and make overlaps and Hamiltonian matrix
elements tractable.

The paper explicitly contrasts this upper-triangular tensor with a full
antisymmetric tensor. Even a two-electron determinant is rank two in the full
antisymmetric matrix but rank one in the ordered representation. Its numerical
Schmidt spectra show slow decay for the full antisymmetric tensor and much
better behavior for HS-MPS.

The one-site variational algorithm factors electronic Hamiltonian matrix
elements through complementary operators. A sweep scales as
`O(N(K^3 D^3 + K^4 D^2))`, one factor of `N` above the corresponding
Fock-space DMRG estimate in the paper.

## Relation to FEMPS

HS-MPS is dangerous prior art for any broad claim about electron-indexed MPS,
avoiding antisymmetric particle-TT rank, or prefix/suffix correlation spaces.
It is also independent confirmation of the project's direct-antisymmetric-TT
no-go motivation.

The candidate FEMPS distinction must be stated precisely: HS-MPS stores the
ordered orbital-string/CI vector and lets determinant/creation-operator algebra
carry signs; FEMPS proposes a continuous first-quantized functional-basis state
in `Lambda^N V_D` built through matrix-wedge products. This distinction alone
does not prove a contraction advantage.

