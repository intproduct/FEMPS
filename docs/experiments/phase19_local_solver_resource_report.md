# Phase 19 interim report: chi-32 local-solver resource closure

Status: completed interim checkpoint. The final Gate G decision is in
`phase19_n8_refinement_gate_report.md`.

## Outcome

The Gate F chi-32 local-DMRG resource qualification is **closed**. The previous
five-operand effective-Hamiltonian einsum requested a 78.12 GiB CUDA
intermediate at `N=8,D=10,chi=32,W=128`. Replacing it by four explicit
two-operand contractions reduces the analytically bounded largest staged
temporary to 13,107,200 float64 entries (100 MiB).

This is an interim Phase 19 result, not Gate G. MPO tangent-gradient, basis,
and external-reference checkpoints remain open.

## Contraction ordering

For environments `L[p,q,r]`, `R[u,t,w]`, local MPO tensors
`Wi[q,s,a,b]`, `Wi1[s,t,c,d]`, and two-site state `Theta[r,a,c,w]`, the old
single expression was

```text
einsum("pqr,qsab,stcd,utw,racw->pbdu", L, Wi, Wi1, R, Theta).
```

The new path is

```text
X[p,q,a,c,w] = einsum("pqr,racw->pqacw", L, Theta)
Y[p,s,b,c,w] = einsum("pqacw,qsab->psbcw", X, Wi)
Z[p,t,b,d,w] = einsum("psbcw,stcd->ptbdw", Y, Wi1)
out[p,b,d,u] = einsum("ptbdw,utw->pbdu", Z, R).
```

Every temporary is bounded by `chi_left*chi_right*W*D^2`. A direct random
five-operand reference agrees with the staged result, and all existing
effective-Hamiltonian, matrix-free Lanczos, and DMRG tests pass.

## Formal Blackwell audit

The formal script does not use the ignored Phase 18 checkpoint. It reproduces
the fixed Phase 18 best blind seed 1862 from scratch using the four-stage AD
schedule, then rebuilds the same structured bond-128 MPO and resets CUDA peak
statistics before DMRG.

```text
particles                 8
basis order               10
MPS bond                  32
MPO bond                  128
Fourier/local quadrature  96 / 192
local Lanczos             30 iterations, one restart
sweeps                    right + left
peak-memory budget        2 GiB
sweep-consistency budget  1e-6
```

The reproduced AD energy is `44.45437326357539`. The right and left sweep
energies are `44.45432190605033` and `44.45432190601349`, differing by
`3.68e-11`. Maximum local discarded weight is `2.20e-12`. The measured DMRG
peak is 736,884,736 bytes (`702.7 MiB`), and the two sweeps take 27.8 seconds.
All predeclared memory, nonincrease, consistency, and no-raw-MPO checks pass.

## Scope

The result repairs the contraction-path resource bug and admits chi-32 local
optimization for this controlled N=8 point. It does not close Gate G: the Gate
F auxiliary raw-gradient miss still requires a gauge-aware directional audit,
and N=8 basis/operator/reference refinement remains incomplete.

## Reproduction

```powershell
.\.venv\Scripts\python scripts\benchmark_phase19_dmrg_memory.py --device auto
```

The machine-readable record is
`results/phase19_dmrg_contraction.json`.
