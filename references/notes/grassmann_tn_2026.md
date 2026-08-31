# Kong, Zhu, and Xie 2026 - Grassmann tensor networks

## Scope

This review constructs fermionic tensor networks through coherent states and
Grassmann variables. Fermionic statistics are encoded locally by Grassmann
parity/order, and the formalism is shown equivalent in common settings to
Z2-graded tensors and fermionic swap gates. It develops Grassmann MPS/MPO,
DMRG, TEBD, CTMRG, and imaginary-time evolution.

For state MPS, the ordinary coefficient tensor is an occupation-number tensor
over single-particle Fock sites. It is first factorized as an ordinary MPS;
Grassmann variables and parity constraints are then attached to physical and
virtual indices. This is not an electron-coordinate particle-site tensor and
not the 2201 first-quantized functional coefficient tensor.

## Relation to FEMPS

Grassmann algebra is valid prior art and a plausible backend for managing
signs, auxiliary integrals, and exterior transfer rules. It cannot be claimed
as the conceptual novelty of FEMPS. Conversely, using a standard Grassmann
MPS over occupation sites would violate the project's no-second-quantization
red line and should be labeled a comparison/backend rather than the FEMPS
state definition.

