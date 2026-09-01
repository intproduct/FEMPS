# ADR 0028: One combined manuscript until a distinct FEMPS method exists

## Status

Accepted on 2026-09-02.

## Context

The structural/no-go manuscript establishes exchange-rank and contraction
constraints. A later restricted-method draft collected results for the
nonbranching diagonal-path implementation. That implementation is exactly a
finite nonorthogonal Slater expansion contracted through determinant
transitions. Its current numerical value is real, but it is scientifically
NOCI-equivalent and has not demonstrated explicit-correlation `D`-convergence
benefits or a matched advantage against Li--Waintal or same-basis DMRG.

Treating this evidence as an independent FEMPS method paper would make the
publication claim outrun the demonstrated novelty.

## Decision

Maintain one combined structural/no-go manuscript. Incorporate current
restricted-solver results only as a numerical validation and algorithm-design
consequence. Freeze the standalone restricted-method draft as an internal
working note; do not develop, submit, or describe it as Paper B.

Restore explicit visibility of the original structural Theorems 1--3. State
the bond boundary exactly: the hard amplitude carrier has `chi=2`, the proved
signed exact-norm reduction has maximum bond three, and a `chi=2` exact-norm
hardness statement remains a conjecture.

## Future-paper gate

A separate method paper requires a non-NOCI differentiator supported by a
matched reproducible benchmark. Admissible routes are:

- explicit correlation beyond a finite Slater expansion with a demonstrated
  functional-basis `D`-convergence advantage; or
- a matched Li--Waintal and/or same-orbital-basis DMRG comparison showing a
  clear accuracy, stability, memory, or complexity tradeoff.

Merely adding determinant terms, seeds, basis points, implementation speedups,
or a larger NOCI-equivalent benchmark does not pass this gate.

## Consequences

- The authoritative publication source is
  `math/femps_no_go_manuscript.tex`.
- Current diagonal-path results remain valid numerical evidence, but no
  independent novelty claim is attached to them.
- The active research plan first closes the combined manuscript and then
  audits genuinely non-NOCI carriers/comparators before further paper writing.
- Four-form rank searches remain parked unless they directly decide this
  algorithmic gate.
