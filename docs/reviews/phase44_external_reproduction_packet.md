# Phase 44 external reproduction packet

## Status and requested reproducer

This packet is ready for an independent external researcher or numerical-
methods group. No external reproduction has yet been received. Repository-
level clean reruns and an independent in-repository verifier are not described
as external scientific replication.

## Immutable checkpoints

- ADR and optimizer contract: commit `61a7550`;
- reference-free initialization fixture: commit `2ce6f9d`;
- frozen production runner: commit `c32b9b3`;
- complete failed result and raw samples: commit `39414e2`.

The authoritative remote is `https://github.com/intproduct/FEMPS`. Reproduce
from commit `39414e2` or a later descendant that leaves every source hash in
the primary artifact unchanged.

## Scientific statement to reproduce

The full Phase 44 gate **fails**. The low-D physical subgate passes at
consecutive `D=(4,6)`, and all held-out confirmation, symmetry, monotonicity,
and resume checks pass. The overall failure is caused by:

- D4 selection standard errors above `2.5e-4`; and
- D6/D8 selection ESS values below 50,000.

An external report must retain both the passing subresult and the failed
aggregate result. It must not call Phase 44 a gate pass.

## Committed-data verification

The original environment was Python 3.12.13, PyTorch 2.11.0+cu128, NumPy
2.3.5, CPU float64. GPU is not used. After installing the pinned project and
sibling latticeTN dependencies, run:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python scripts/verify_phase44_initialization_fixture.py
python scripts/verify_phase44_n4_explicit_correlation_d_gate.py
python -m pytest tests/test_phase44_n4_explicit_correlation_artifact.py -q
```

Expected decisions are:

```text
phase44_interacting_d_gate_pass = false
two_consecutive_D_advantage_pass = true
consecutive_advantage_pairs = [[4,6]]
all_confirmation_gates_pass = true
maximum_observable_difference = 0
```

## Clean external production

Use new output and checkpoint directories; do not reuse the committed local
checkpoints or overwrite repository evidence:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python scripts/benchmark_phase44_n4_explicit_correlation_d_gate.py `
  --ledger external-reproduction/phase44_pre_reference_selection.json `
  --output external-reproduction/phase44_result.json `
  --archive-dir external-reproduction/samples `
  --checkpoint-dir external-reproduction/checkpoints
```

The run requires roughly 4 GB peak process RSS and about 35 minutes on the
original workstation. Preserve the complete external output, all three sample
archives, every optimizer/evaluation checkpoint, Python package lock or
environment report, host/CPU/thread information, and console log. Do not
change a seed, budget, proposal scale, threshold, initialization, lineage, or
axis after observing results.

Then compare the clean result with the authenticated primary artifact:

```powershell
python scripts/compare_phase44_external_reproduction.py `
  --reproduction external-reproduction/phase44_result.json `
  --output external-reproduction/phase44_comparison.json
```

The comparator authenticates the primary JSON against its committed manifest,
checks the frozen design and source hashes, loads the reproduction's own six
optimizer checkpoints plus its D6 clean control, and recomputes all 12
selection/confirmation observables from the reproduction's raw coordinate
archives. It requires identical gate decisions and selected lineages. Combined
energies are compared with the preregistered uncertainty allowance
`5 sqrt(SE_primary^2 + SE_reproduction^2) + 2e-4`, rather than a cross-machine
bitwise-equality demand. It also checks the `1e-12` antisymmetry tolerance and
absence of forbidden tensor/path materialization.

`numerical_reproduction_pass = true` means only that the supplied numerical
artifacts pass these checks. The script deliberately leaves
`external_independent_replication_complete = false`: software cannot establish
the reproducer's identity, independence, conflicts, or non-reuse of the
primary checkpoints. Those require the named-human report below.

## Primary hashes

| object | SHA-256 |
|---|---|
| D4 sample archive | `aa3e27025fc44ce0ccdc265a90b63f046e04b9bba5145f31c608e5799eaed546` |
| D6 sample archive | `336cc4391c93584314ee0e7ba1170ce59cf94c7f1cebfd7b80cdb9b09b5605d1` |
| D8 sample archive | `9313e542459a1d7b596aa567d113887c7866301139c42d52967e8830c3ca3637` |
| pre-reference ledger (normalized text) | `e3f9937c88c792d4386ae3b69beab92440c7d4ba99d33e1ddb747b83a297e9cf` |
| production runner (normalized text) | `eabe2697f69dd4e08d7126a59c97292d8a550f26cfaac2600d8ed775766ae1e5` |
| optimizer backend (normalized text) | `72769e489a3226597bf0802e439a2d0673650a8e8e9e46d13a23bdfaaa86d4af` |
| fixed-state backend (normalized text) | `deffaaa8af24efa10ac25b35417d410cc83062bf3d155d0ce3838d0e2035cab4` |
| ADR 0033 (normalized text) | `0943aa2e391ee4fe4592a671cdacbcd163266ce2703e22a77387d87781931a92` |
| initialization fixture (normalized text) | `42327353c878891682e5d3738a51db34b828e54950d17447fd201d2f95451d9c` |

Seven optimizer-checkpoint hashes are recorded separately in
`docs/experiments/results/phase44_optimizer_checkpoint_manifest.json`.

## Required external report

Record reproducer name, affiliation, conflicts, repository commit, environment,
hardware, thread controls, wall time, peak memory, all output hashes, any
deviation, every frozen gate, and whether the aggregate failure plus `(4,6)`
subgate are reproduced. Attach raw artifacts or a durable repository link.

External reproduction does not by itself prove ansatz novelty, scalability,
or superiority. It also does not convert an occupation-number MPS into FEMPS
or authorize Paper B without a separate publication decision.
