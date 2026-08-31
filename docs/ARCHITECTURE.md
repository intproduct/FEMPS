# Architecture

```mermaid
flowchart LR
    Basis[Functional basis] --> Operators[Functional operator matrices]
    Operators --> Hamiltonian[Continuous Hamiltonian]
    Basis --> Exterior[Exterior algebra]
    Exterior --> State[FEMPS state]
    Lattice[latticeTN AD/MPS backend] --> Baseline[2201 baseline]
    Hamiltonian --> Baseline
    Hamiltonian --> Contract[Exterior contraction engine]
    State --> Contract
    Contract --> Optimizer[AD optimizer]
    Reference[Small full tensor / exact solver] --> Validation[Validation]
    Baseline --> Validation
    Optimizer --> Validation
    Certificates[Math certificates] --> Theory[Theorem artifacts]
```

## Module boundaries

- `femps.basis`: continuous orthonormal bases and their operator matrices.
- `femps.baselines`: reproducible non-fermionic functional-TN baselines.
- `femps.exterior`: wedge/forms/reference materialization and the conditionally
  admitted fixed-number Pfaffian contraction engine.
- `femps.states`: Slater, explicit antisymmetric, and FEMPS states.
- `femps.algorithms`: contractions and optimization admitted after exact tests.
- `math/`: proof and certificate pipelines, isolated from production solvers.

`latticeTN` stays an upstream sibling dependency. FEMPS reuses its PyTorch MPS,
native contractions, device/dtype conventions, and AD infrastructure instead
of copying them.
