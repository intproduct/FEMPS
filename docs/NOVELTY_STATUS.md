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
