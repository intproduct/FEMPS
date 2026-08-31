# ADR 0001: keep statistics outside ordinary particle TT

## Status

Accepted as the research direction. Gate A subsequently admitted only the
fixed-number Pfaffian/AGP structured subclass; see ADR 0002.

## Decision

Ordinary particle-site TT is retained only as a baseline/no-go control. The
main ansatz will enforce fermionic exchange through an exterior structural
layer. Grassmann, determinant, Pfaffian, or occupation-space tools may be used
as contraction backends, but they do not redefine the state as a
second-quantized MPS.

## Consequence

The repository develops exact small-N exterior references before scalable
algorithms. The conditional Gate A result activates the Pfaffian structured
subclass; generic matrix-wedge code remains a small-system oracle rather than
an undocumented exponential solver.
