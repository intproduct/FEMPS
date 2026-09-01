# Completed execution plan: Phase 42 clean-source reproduction of Phase 40

## Objective

Test deterministic end-to-end reproducibility of the only admitted non-NOCI
differentiator without reusing any Phase 40 tensor checkpoint. This is a clean
optimization reproduction, not an external research-group replication and not
authorization for Paper B.

## Frozen reproduction

- use the exact Phase 40 model, `D/P/K` axes, seeds, optimizer budgets,
  quadrature orders, selection rule, and thresholds;
- write to a new artifact and a new checkpoint directory;
- do not read Phase 40 checkpoints or initialize from Phase 40 optimized
  tensors;
- compare every serialized energy, norm, variance, residual, seed selection,
  point decision, and final gate decision with the primary artifact;
- require maximum deterministic observable difference at most `2e-10` and an
  identical set of consecutive passing `D` pairs;
- run the independent state reconstruction verifier on the new artifact;
- do not add seeds, rescue steps, or alternative `P/K` choices if a mismatch
  occurs.

## Interpretation boundary

A pass establishes repository-level clean-source reproducibility. It does not
establish independent scientific replication, many-particle scalability, or
novelty beyond established explicit-correlation/Jastrow methods. A Paper-B
decision remains closed until a genuinely independent reproduction and a
many-particle controlled-contraction/comparator result exist.

## Completion record (2026-09-02)

- reran all 72 correlated and 54 NOCI optimizations in a new checkpoint tree;
- did not read or reuse any primary Phase 40 optimized tensor;
- obtained zero difference for every compared energy, variance, norm,
  residual, and quadrature uncertainty;
- reproduced the passing pairs `(2,4)`, `(4,6)`, `(6,8)` and the failures at
  `D=10,12` exactly;
- independently reconstructed both full artifacts and retained the explicit
  boundary that external scientific replication is still incomplete.
