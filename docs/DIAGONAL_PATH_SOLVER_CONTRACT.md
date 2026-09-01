# Diagonal-path FEMPS solver and reproduction contract

Contract version: 2 (Phase 34 compatible extension, 2026-09-01)

## Scientific scope

The public solver implements the restricted nonbranching diagonal-path FEMPS

\[
  \Psi(x_1,\ldots,x_N)=\sum_{a=1}^{K} c_a\,
  u_{a1}\wedge\cdots\wedge u_{aN}.
\]

Each orbital is represented in a continuous functional basis of order `D`.
This is a first-quantized particle-coordinate state and an exact subclass of
matrix-wedge FEMPS. It is not a generic matrix-wedge contraction and is not an
occupation-number or second-quantized MPS.

## Frozen public Python API

The following names are exported from `femps.algorithms` for the Phase 28/29
reproduction line:

- `DiagonalPathConfig`;
- `canonical_slater_orbitals`;
- `embed_diagonal_path_orbitals` for exact nested `D` growth;
- `extend_diagonal_path_terms` for exact blind nested `K` growth;
- `select_adaptive_diagonal_path_term` for seeded, truth-free, one-term greedy
  growth through fixed-span determinant-transition energies;
- `run_diagonal_path_variable_projection`;
- `load_diagonal_path_checkpoint`;
- `validate_diagonal_path_checkpoint` and `validate_diagonal_path_result`;
- `DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION` and
  `DIAGONAL_PATH_RESULT_SCHEMA_VERSION`.

Fields may be added compatibly, but existing names, meanings, tensor ordering,
or units cannot change without a schema increment and migration note. The
orbital tensor order is always `(K,D,N)`.

## State, operator, and optimization contract

- QR-gauged columns define each Slater term.
- Linear amplitudes are eliminated through a conditioned Hermitian generalized
  eigenproblem at every nonlinear step.
- One-body matrices use functional-basis indices `(p,q)`.
- A factorized two-body operator is an explicitly particle-exchange-symmetric
  representation of the physical `(p,q,r,s)` tensor. Its backend, retained
  rank, threshold, and dense reconstruction error must be recorded by formal
  benchmarks.
- User-provided operators require a nonempty `operator_id`. Resume rejects a
  changed configuration, schema, or operator identity.
- Seeds, optimizer step counts, learning rates, truth/materialization caps,
  dtype, device, and checkpoint lineage are part of reproduction evidence.

Adaptive growth may rank a fixed seeded Slater pool using only the current
state's factorized determinant-transition Hamiltonian, overlap matrix,
generalized-eigenvalue energy, and balanced conditioning. It may not read a CI
energy, CI vector, dense exterior Hamiltonian, or materialized particle tensor.
The selected `K+1` span contains the source K-term span exactly, and every
candidate decision records the predicted energy, improvement, retained rank,
condition number, and rejection reason. Dense CI is permitted only after term
selection and nonlinear optimization as a final truth audit.

## Checkpoint schema v1

Checkpoints contain configuration, operator identity, current/best `(K,D,N)`
orbitals, best energy, optimizer/scheduler state, step, and history. Formal
scripts load them only through `load_diagonal_path_checkpoint`; direct
`torch.load` is not an admitted reproduction path.

Checkpoints are ignored runtime artifacts. A committed result must include the
commands and lineage needed to regenerate them. Bitwise equality across
different Torch/CUDA stacks is not promised; registered numerical tolerances
and independent physical invariants define reproduction.

## Result schema v2

Every solver call reports at least:

- method and numerical-evidence label;
- full configuration, environment, operator metadata and identifier;
- completion/initialization status and optimization history;
- energy, norm error, energy variance when bounded exterior truth is enabled,
  and generalized-eigenproblem conditioning;
- structural antisymmetry residual on every call;
- materialized antisymmetry residual whenever `D**N` is within the configured
  materialization cap; otherwise the field is explicitly `null`;
- structural operation counts, including zero enumerated virtual paths;
- sampled process peak RSS, elapsed time, and CUDA peak allocation when used;
- finite-basis truth and ordinary particle-TT ranks when their declared caps
  permit those validation-only calculations.

`validate_diagonal_path_result` enforces the stable required fields, evidence
label, method identity, materialization rule, memory record, and zero-path
condition before the public solver returns.

## Complexity and forbidden work

For `K` orbital matrices of shape `D x N` and two-body factor rank `L`, the
well-conditioned production path has `K^2` determinant transitions and a
two-body cost polynomial in `K^2 L (D^2 N + N^3)`. Stored state parameters are
`O(KDN)`. Singular-safe minors are a polynomial validation/fallback path.

Production may not enumerate `K^(N-1)` virtual paths or materialize the full
`D^N` particle tensor. Full exterior coefficients, dense CI, and particle
materialization are explicitly bounded truth tools and must be labeled as
such.

## Scientific failure rules

- Structural antisymmetry residual is never omitted.
- Approximation, truncation, or stochastic uncertainty must be explicit; this
  exact restricted route currently uses no stochastic contraction estimator.
- Floating-point outputs are numerical evidence, not theorems.
- No efficiency, scalability, or superiority claim follows from a passing
  small-system artifact alone.
- Failure to meet a registered gate is recorded without adding seeds,
  dimensions, or mathematical searches merely to hide the failed point.
