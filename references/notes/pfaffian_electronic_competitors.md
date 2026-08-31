# Pfaffian and first-quantized electronic competitors

## Bajdich et al. 2008

Electronic-structure QMC already uses singlet/triplet Pfaffian pair orbitals,
linear combinations of Pfaffians, and backflow. This is the most direct warning
against presenting the Pfaffian carrier or multi-Pfaffian expressivity as new.
Its numerical engine is real-space sampling and nodal optimization, whereas the
current FEMPS subclass uses deterministic finite functional-basis contractions.

## FermiNet and PauliNet

Both are continuous first-quantized electronic Schrödinger solvers with exact
antisymmetry carried by determinant structures and correlations supplied by
neural coordinate dependence/Jastrow/backflow. They are much more expressive
and physically realistic than the present 1D soft-Coulomb benchmark. FEMPS must
not claim first-quantized priority, basis-free accuracy, or current superiority.

## Hidden-fermion Pfaffian state

The 2025 HFPS preprint combines a Pfaffian with neural hidden fermions and
reports scalable interacting-fermion simulations, primarily for lattice
Hubbard settings. Its domain and correlation mechanism differ, but it narrows
any broad claim about scalable correlated Pfaffian wavefunctions.

## Deterministic linear combinations of AGPs

Uemura, Kasamatsu, and Sugino (2015) already formulate configuration
interaction as a linear combination of independently optimized AGPs/HFB states
and report polynomial cost with quadratic dependence on the number of terms.
Dutta et al. (2021) construct linearly independent nonorthogonal AGP sets and
use them for selective CI. Kawasaki, Gao, and Scuseria (2026) further rewrite
inter-geminal AGP-CI as compact LC-AGP expansions using border-rank ideas.

These works overlap the present finite-AGP state family, generalized amplitude
solve, and `K^2` transition-matrix organization much more directly than the QMC
comparators. Gauge-balanced conditioning and the soft-Coulomb `D/K` scans are
useful numerical engineering, but they do not establish a new ansatz class.

## Surviving narrow question

The potentially distinct package must now be stated more strictly: a 2201
particle-coordinate functional solver with new exterior-transfer/canonical
structure beyond standard LC-AGP, or a successful generic matrix-wedge
statistics/multiplicity factorization. The current finite-AGP solver remains a
valuable polynomial fallback and benchmark control, but it is not sufficient
as the central novelty claim.
