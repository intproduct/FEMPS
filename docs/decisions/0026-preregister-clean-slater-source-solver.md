# ADR 0026: Preregister the clean Slater-source solver command

- Status: accepted before implementation and production
- Date: 2026-09-01
- Depends on: ADR 0025 and accepted Phase 36

## Context

Phase 36 exposes a bounded adaptive API, but its admitted N6 reproduction
starts from a historical preoptimized K4 checkpoint.  That is a valid API
reproduction and not yet an end-to-end physical solver.  Phase 37 must start
from a state constructed solely from explicit model/configuration inputs.

A bounded exploratory audit, performed before this ADR and not admitted as a
production artifact, used the existing N4,D6 soft-Coulomb truth region.  The
canonical lowest-orbital Slater followed by the seed schedule frozen below
gave nonincreasing K1--K4 energies and a final same-basis CI error below
`5e-10`.  Those numbers select a reproducibility configuration; they are not a
blind accuracy claim.  The production gates below are deliberately looser
than the audit and the already accepted Phase 28 N4 controls.

## Decision

Add a public command-level orchestration layer which:

1. builds the harmonic-oscillator functional basis and physical-SVD
   soft-Coulomb operator from a versioned explicit configuration;
2. constructs the canonical K1 determinant from the lowest N basis orbitals;
3. optimizes that source through the existing public single-K solver;
4. passes the accepted source orbitals to the Phase 36 bounded adaptive API;
5. atomically checkpoints command state and supports resume without any
   historical FEMPS artifact; and
6. writes a validated result containing K1 and every requested adaptive stage.

The command may use dense exterior coefficients and CI only after the
variational sequence is frozen, as bounded validation.  Production contraction
must not materialize a labelled-particle `D^N` tensor or enumerate virtual
paths.

## Frozen production configuration

The machine-readable source of truth is
`docs/experiments/configs/phase37_n4_d6_k4.json`:

- spin-polarized `N=4,D=6` harmonic trap;
- soft-Coulomb `g=a=omega=1`, Q128 physical-SVD factorization;
- canonical lowest-orbital K1 source, source seed 3701;
- 60 Adam steps, 40 L-BFGS refinement steps for K1 and each later K;
- mandatory external `max_K=4`, pool size 32;
- K2 seeds 3711/3712, K3 seeds 3721/3722, K4 seeds 3731/3732;
- CPU complex128, overlap threshold `1e-10`, condition cap `1e8`;
- explicit output and command-checkpoint paths.

The registered production is interrupted after the first adaptive stage at K2
and resumed to K4.  A separate clean uninterrupted run uses the identical
configuration and a different checkpoint/output path only.  No seed,
threshold, optimizer, model, N, D, or maximum-K change is allowed after either
run starts.

## Acceptance gates

- Both runs start from the canonical lowest-orbital Slater and use no
  historical FEMPS checkpoint or CI initializer.
- Resume preserves the K2 record and all selected candidate indices; clean and
  resumed energies agree at every K within `1e-11`.
- Energy is nonincreasing through K1--K4 within `1e-9`.
- K1 same-basis CI error is at most `2e-3` and variance at most `1e-2`.
- Final same-basis CI error is at most `1e-6` and variance at most `1e-5`.
- Every K reports norm error at most `1e-10`, structural antisymmetry residual
  at most `1e-12`, zero virtual-path enumeration, and zero production `D^N`
  materialization.
- Operator factorization error is at most `1e-11`; each optimization stage is
  at most 120 seconds, the command is at most 600 seconds, and sampled peak RSS
  is at most 2 GiB.
- An independent verifier reconstructs source/operator identities, exterior
  norms, energies, variances, CI error, ordinary particle-TT ranks, and the
  comparison with the existing manually orchestrated Phase 28 N4 control.
- Small-system materialization, AD, invalid-config, changed-identity, and
  interruption/resume tests pass before production is admitted.

## Failure rule

Any failed gate is reported without changing seeds, thresholds, N, D, or K.
Failure means that the public adaptive API still requires an expert-provided
correlated source; it does not authorize automatic stopping, N8, a stochastic
claim, or renewed high-dimensional form-rank search.

## Scientific boundary

Passing this ADR establishes a reproducible end-to-end command for the exact
restricted nonbranching FEMPS subclass in one small interacting continuum
truth region.  It does not establish generic FEMPS contraction, automatic
stopping, asymptotic scalability, runtime superiority, or determinant-state
novelty.
