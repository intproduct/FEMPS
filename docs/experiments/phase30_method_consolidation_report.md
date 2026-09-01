# Phase 30 FEMPS method consolidation report

## Outcome

Phase 30 converts the accepted restricted diagonal-path FEMPS results into an
externally auditable method package. It does not add a new numerical benchmark
or broaden the admitted solver scope.

## Reproduction manifest

`results/phase30_reproduction_manifest.json` contains seven admitted numerical
artifacts. For each it records a bounded claim, commands, seeds, tolerances,
schema/evidence labels, scientific boundary, and artifact SHA-256.

The independent verifier checks the public solver/checkpoint schema, all hashes
and labels, command targets, unique claim identifiers, and every registered
artifact verifier. It returns 7/7 pass without rerunning long optimizations.

## Claim control

`docs/METHOD_CLAIM_EVIDENCE_MATRIX.md` separates proved structure, bounded
N2/N4/N6 numerical evidence, measured exchange/correlation and cost tradeoffs,
and unestablished speed, asymptotic, N8, generic-contraction, and novelty claims.

ADR 0020 defers second-quantized DMRG. At current `D=10`, the exact exterior
spaces have dimension 210 and direct CI is more informative than an approximate
DMRG reproduction of the same finite-basis Hamiltonian.

## Paper figures

Two 300-dpi PNG/PDF figure pairs are generated only from manifest-hashed data:

1. `femps-convergence-summary`: N4 K/D convergence and N6 K recovery/stability;
2. `femps-structure-cost-summary`: matched N4/N6 value-gradient timing and
   ordinary particle-TT center ranks at fixed FEMPS K.

The provenance sidecar records the manifest hash, source hashes, exact plotted
data, output hashes, evidence label, and claim boundary. Its independent
verifier checks all four figure files.

## Decision

The restricted method package is ready for manuscript drafting and external
review at its stated scope. It establishes a nonbranching first-quantized
continuous functional-basis FEMPS subclass with exact antisymmetry, polynomial
determinant-transition contraction, AD optimization, and reproducible
interacting N4/N6 results with systematic K/D behavior.

It does not establish generic FEMPS efficiency, superiority over CI/DMRG, or
N8/asymptotic scaling. Those limits must remain visible in any manuscript.
