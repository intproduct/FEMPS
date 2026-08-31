# Completed execution plan: Phase 7 E5 particle-number scaling

## Objective

Test whether the separation between ordinary particle-TT statistics complexity
and exterior/Pfaffian correlation control persists beyond four particles.

## Completed checkpoints

- [x] E5a: construct six noninteracting spinless harmonic fermions as a
  correlation-bond-one Slater FEMPS and three-pair-channel Pfaffian state.
- [x] Verify energy `N^2/2=18`, ordinary particle-TT ranks
  `(1,6,15,20,15,6,1)`, flat particle Schmidt spectra, and truncation bounds.
- [x] Reproduce E5a by blind polynomial AD on RTX PRO 4000 Blackwell without
  materializing the particle tensor during optimization.
- [x] E5b: build analytic and exterior-sector truth for six interacting
  harmonic fermions over safe `D` and interaction ranges.
- [x] Compare single AGP and greedy finite-AGP growth, reporting `D`, `K`, seed,
  overlap conditioning, and continuum/basis/ansatz errors separately.
- [x] Record time and memory scaling against the E3/E4 results; materialize only
  safe post-training truth tensors.
- [x] Decide whether to continue to larger `N` or begin the ordered-sector
  comparison before E6 soft-Coulomb physics.

## Exit result

E5a reaches energy `18` with ordinary ranks `(1,6,15,20,15,6,1)` and direct
FEMPS correlation bond one. At `N=6,D=10,kappa=0.1`, blind K=1 is reproducible
to a `5.2e-6` seed spread and no-oracle greedy K=2 reaches `4.765e-6` error
against the independent finite-basis truth, with stable restart and overlap
diagnostics. Phase 7 passes its exit criterion; the evidence is summarized in
`docs/experiments/fermion_e5_report.md`.
