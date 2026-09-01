# Novelty status

The novelty audit is active and incomplete. The working name FEMPS is not a
priority claim. Every candidate claim must be checked against the works in
`references/novelty_matrix.md`, especially first-quantized MPS, HS-MPS,
Grassmann/graded tensor networks, symmetry structural/degeneracy decompositions,
and determinant-carrier correlation ansätze.

Phase 12 found a direct overlap that materially narrows the project claim:
Uemura--Kasamatsu--Sugino (2015) already use deterministic linear combinations
of independently optimized AGPs with quadratic term-count cost; Dutta et al.
(2021) and Kawasaki--Gao--Scuseria (2026) further develop nonorthogonal LC-AGP
and AGP-CI constructions. The present finite-AGP implementation is therefore a
validated fallback and benchmark baseline, not a new ansatz class. The active
novelty question is now explicitly beyond LC-AGP: generic matrix-wedge
contraction or a proved statistics-carrier/correlation-multiplicity structure.

Phase 13 rules out the first branch as a generic exact solver, conditionally on
standard permanent hardness, by an explicit tagged Cayley-determinant
reduction. Thus “generic matrix-wedge FEMPS with polynomial exact contraction”
is no longer an admissible novelty claim. A future method claim must instead
identify a systematically improvable restricted algebra with a surviving
novelty boundary, prove a stronger statistics-carrier factorization, or use
the ordered-sector first-quantized route. The one-body quantity
`N/Tr(gamma^2)` is retained only as a correlation diagnostic, not a novel
entanglement measure or a canonical FEMPS bond spectrum.

Post-gate numerical work now targets ordered-sector/interparticle-distance
functional TN and is not called FEMPS. Li--Waintal 2026 already establishes
the first-quantized ordered-distance MPS direction; any surviving contribution
must be stated narrowly as the 2201 continuous orthonormal functional-basis
operator/AD integration, its controlled continuum benchmarks, and the new
matrix-wedge/ordinary-particle-TT no-go theory. No priority claim is attached
to ordered coordinates or distance-variable MPS.

Phase 15 closes only the finite-grid contraction gate: the exact hard-charge
gap MPS, `O(N^2(L-N))` raw operator bond, and native AD optimizer are an
implementation/evidence result, not a new ansatz claim. The next novelty gate
is whether full-line center-of-mass and Dirichlet half-line distance bases can
retain the distinctive 2201 orthonormal functional-operator calculus with
controlled continuum convergence. Until that comparison is complete, the
ordered-distance branch has no affirmative method-priority claim.

Phase 16 closes Gate D at controlled small-system scope. Implementation-level
review confirms that the result is a bridge between, rather than a replacement
for, its two direct parents: Hong et al.'s orthonormal continuous
functional-operator/AD construction and Li--Waintal's ordered-distance
first-quantized MPS. The surviving package is the exact COM/gap calculus,
Dirichlet and unbounded half-line bases, continuum mixed-derivative and
soft-Coulomb MPOs, native global AD, and an independently separated
`D/scale/K/chi/optimization` error budget. This is a reproducible integration
and evidence contribution. It is not an affirmative ansatz-priority claim,
and it is not called FEMPS.

Phase 17 closes Gate E at controlled N=6 scope without broadening that claim.
The implemented Fourier--Bessel/odd-Hermite interaction, compact four-real-state
all-pair recurrence, and global compression audits are an integration and
evidence result. The recurrence removes direct-pair growth at fixed Fourier
order, but no priority claim is made for ordered coordinates, first-quantized
distance MPS, or scalable fermionic tensor networks. Hong et al. and
Li--Waintal remain the direct method parents. The N=6 basis-dominated error and
temporary dense raw-MPO storage preclude an asymptotic accuracy/resource claim.

Phase 18 closes the core Gate F criteria at one controlled N=8 point, again
without broadening the novelty claim. The Lowdin-orthonormalized two-scale
half-line basis and incremental sparse-recurrence MPO builder are basis/
implementation choices inside the established ordered first-quantized and
functional-basis parent methods. They reduce measured error and construction
memory but do not establish a new ansatz class. The retained raw-gradient
auxiliary miss, numerical rather than continuum N=8 reference, and chi-32
local-solver resource rejection explicitly preclude an asymptotic or method-
priority claim.

Phase 19 closes those two operational Gate F qualifications without broadening
the novelty claim. A bounded local effective-Hamiltonian contraction order, a
left-gauge physical-tangent audit, matched MPO-bond training, and a blind D12
refinement are implementation and evidence controls inside the same Hong et
al./Li--Waintal parent route. The exterior D14 value remains a numerical
reference, the N=2/4/6/8 trend does not admit N=10, and the route is still not
called FEMPS. Phase 20 therefore returns to the unresolved novelty boundary: a
restricted exterior correlation structure that is both genuinely beyond
finite LC-AGP/Gaussian families and exactly polynomially contractible, or a
documented negative classification.
