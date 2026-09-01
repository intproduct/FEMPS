# ADR 0009: Resource-safe N=8 Gate G passes without N=10 admission

- Status: accepted
- Date: 2026-09-01

## Context

Gate F admitted one controlled N=8,D=10 ordered-distance functional-TN point
but retained two qualifications. A five-operand chi-32 two-site effective-
Hamiltonian contraction requested a 78.12 GiB temporary, and an intentionally
strict bond-128/192 raw MPS-parameter gradient threshold missed even though
fixed-state energy and gradient direction agreed closely. The exterior N=8
reference was limited to D=12.

Phase 19 replaces the local action with four bounded two-operand contractions.
A formal chi-32 reproduction peaks at 736,884,736 CUDA bytes and two local
sweeps agree within `3.68e-11`. A left-canonical physical-tangent audit removes
gauge-coordinate ambiguity: bond 128 versus 192 differs by `3.94e-8` in the
largest directional derivative, `2.02e-5` in relative L2, and has cosine
`0.999999999811`, passing all declared budgets. The historical Gate F raw
parameter-gradient miss remains unchanged in its original record.

Matched bond-128/160/192 training passes the own-versus-bond-192 energy,
cross-run spread, and 2 GiB memory budgets, selecting bond 128 as the smallest
passing production bond. The independent exterior reference is extended to
D=14, where Q128 and Q160 differ by only `9.24e-13`; it remains numerical.

A blind D12 multiscale scan selects `(ell,rho)=(0.55,3.0)`. Its production
energy `44.4528442233` differs from the exterior D14 Q160 reference by
`7.174e-3`, a `17.6%` reduction from Gate F D10 against that same reference.
Fourier, local quadrature, independent chi-32 optimization, basis
orthonormality/conditioning, and memory controls pass.

## Decision

Accept Gate G as a resource-closed controlled N=8 ordered-distance
functional-TN point.

Acceptance conditions are:

1. use the staged latticeTN effective-Hamiltonian action for admitted chi-32
   local optimization and retain its dense small-system regression;
2. qualify MPO gradients through gauge-fixed, many-body-normalized physical
   tangent directions rather than raw tensor coordinates;
3. retain the Gate F raw-gradient miss as historical evidence rather than
   retroactively changing its threshold;
4. retain bond 128 as the smallest passing production MPO bond and bond 192 as
   an audit reference;
5. keep D14 values labeled finite-basis numerical references, not continuum
   bounds;
6. preserve scale, scale ratio, D, Fourier order, local quadrature, MPO bond,
   MPS bond, seed, and schedule as independent controls;
7. do not admit N=10 from the present N=2/4/6/8 descriptive trend; and
8. continue to call this route ordered-distance functional TN, not FEMPS.

## Consequences

- Both explicit Gate F qualifications are operationally closed at the
  controlled N=8 point.
- N=8,D=12 becomes the strongest admitted ordered-distance numerical point.
- The phase does not establish an asymptotic law, a continuum certificate, or
  method priority.
- Further automatic particle-number scaling is deferred. The next phase
  returns to the central exterior-algebra question: a tractable restricted
  correlation structure beyond known finite LC-AGP, or a documented negative
  result.
