# Phase 26 paired-state and fermionic-entropy audit

## Scope

This note records the literature boundary added for manuscript v1. It does not
claim a complete review of projected quasiparticle methods or particle
entanglement.

## Paired-state contraction

- Khamoshi--Henderson--Scuseria give efficient AGP reduced-density-matrix
  evaluation. This directly precedes the project's fixed-number AGP control
  contractions; those formulas are not FEMPS novelty.
- Dukelsky--Pittel--Esebbag analyze number-projected BCS structure. Number
  projection and the AGP/projected-BCS relationship are established rather
  than a new exterior construction.
- Robledo resolves the sign of HFB overlaps through a Pfaffian formula. The
  project must not present Pfaffian overlap/sign handling as new.
- Tahara--Imada combine quantum-number projection, pair-product/Pfaffian trial
  states, and many-variable variational Monte Carlo. Projected Pfaffian
  optimization is therefore a direct prior-art comparator even though its
  stochastic Fock-space setting differs from the 2201 functional basis.
- Bajdich et al. already provide continuous real-space Pfaffian pairing and
  multi-Pfaffian QMC controls.

## Particle entanglement

Carlen--Lieb--Reuvers establish rigorous entropy and entanglement bounds for
fermionic reduced density matrices. The manuscript may use the elementary
explicit Slater particle-Schmidt decomposition and its ordinary-TT consequence,
but must not imply that particle entanglement, Slater extremality, or its bounds
were introduced here. The contribution-Gram diagnostic remains an
expansion-dependent correlation diagnostic, not a new entanglement entropy.

## Manuscript consequence

The paired/Pfaffian families are attributed polynomial controls or competitors.
The Phase 26 project-specific statements are the transfer of established
noncommutative-determinant and permanent results into explicit FEMPS
contraction obstructions, together with the 2201-specific representation
motivation and exact evidence package.
