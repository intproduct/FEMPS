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
