# Novelty matrix (v0)

| Work | Quantization | Site meaning | Antisymmetry | Continuous functional basis | State ansatz | Contraction | Overlap with FEMPS | Open distinction |
|---|---|---|---|---|---|---|---|---|
| Hong et al. 2022 | first | particle coordinate | deferred to fermionic TN | yes | ordinary coefficient MPS | standard MPS | direct functional-basis parent | exterior carrier and fermionic solver absent |
| Li-Waintal 2026 | first | ordered particles, encoded by interparticle distances | retain one ordered sector; recover others by permutation | coordinate grid, not the 2201 orthonormal basis layer | distance-variable MPS | low-rank MPO with constraint/cutoff | directly removes exchange multiplicity | FEMPS full-space exterior carrier vs ordered-sector route |
| Li-Chan 2016 HS-MPS | fixed-N Hilbert space | electron sites with ordered orbital indices | strictly upper-triangular independent CI coefficients; determinant algebra carries signs | finite orbital basis, not a direct continuous solver | prefix/suffix-constrained HS-MPS | complementary-operator DMRG | electron-indexed chain and direct-antisymmetric rank diagnosis | Pfaffian functional exterior subclass is conditionally admitted; generic matrix-wedge remains uncontracted |
| Kong-Zhu-Xie 2026 Grassmann TN | Fock/coherent-state | occupation sites and graded virtual bonds | Grassmann parity/order, equivalent to Z2 grading/swap gates | no | Grassmann MPS/PEPS/MPO | Grassmann integration with ordinary coefficient numerics | possible sign/contraction backend | not a first-quantized particle-coordinate ansatz |
| Beylkin et al. 2008 | first | continuous particle coordinates | Slater determinant sum | continuous functions | unconstrained Slater sum | determinant identities | determinant statistics carrier | matrix-wedge chain and AD functional operators |
| Begovic Kovac--Kressner 2017 | coefficient tensor | shared one-particle Tucker modes | alternating core retained | finite basis | structure-preserving Tucker approximation | HOOI/Jacobi-style optimization | rigorous symmetry-preserving compression comparator | one-particle support rank, not particle-cut TT rank |

All entries are preliminary reading notes, not novelty conclusions.
