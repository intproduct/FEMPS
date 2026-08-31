# Soft-Coulomb functional operator

## Model convention

Phase 9 uses spin-polarized one-dimensional fermions on the full real line,
with oscillator units `hbar=m=omega=1` and Hamiltonian

\[
H=\sum_i\left(-\frac12\partial_{x_i}^2+\frac12x_i^2\right)
+g\sum_{i<j}\frac{1}{\sqrt{(x_i-x_j)^2+a^2}}.
\]

The initial benchmark fixes repulsive `g=1` and softening length `a=1`. This is
an electronic-like one-dimensional model, not a realistic Coulomb calculation.
All particles have the same spin, so the spatial wavefunction is strictly
antisymmetric. The trap replaces box boundary conditions by square-integrable
decay on `R`.

## HO-basis quadrature

For unit-frequency normalized oscillator functions
`phi_n(x)=exp(-x^2/2) p_n(x)`, the two-body tensor is

\[
V_{pqrs}=g\int dx\,dy\,
\phi_p(x)\phi_r(x)\frac{1}{\sqrt{(x-y)^2+a^2}}
\phi_q(y)\phi_s(y).
\]

Gauss--Hermite nodes and weights apply directly after removing the two Gaussian
factors. The normalized polynomials `p_n` are evaluated by a three-term
recurrence rather than by factorials or raw high-degree Hermite coefficients.

## Symmetric factorization

Let `F_i[p,r]=p_p(x_i)p_r(x_i)` and define the weighted kernel

\[
M_{ij}=\sqrt{w_i}\,
\frac{1}{\sqrt{(x_i-x_j)^2+a^2}}\sqrt{w_j}.
\]

If `M=U diag(lambda) U^T`, then

\[
O_l[p,r]=\sum_i\sqrt{w_i}F_i[p,r]U_{il},\qquad
V=\sum_l g\lambda_l O_l\otimes O_l.
\]

This is exactly the symmetrized `FactorizedTwoBodyOperator` interface used by
the polynomial Pfaffian contractions. Eigenmodes may be removed only by an
explicit relative threshold. Every construction reports retained/discarded
rank, the largest discarded kernel eigenvalue, dense four-index reconstruction
error, Hermiticity residual, and particle-exchange residual.

## Independent checks and evidence limits

The implementation retains a direct four-index quadrature sum that does not use
the kernel eigendecomposition. Small-system tests compare this tensor with the
factorized reconstruction, then compare polynomial AGP energy and gradients
with an explicitly materialized exterior Hamiltonian.

Quadrature convergence, operator factorization, HO-basis convergence, and
variational optimization are distinct error sources. A largest computed `D`
value is only a numerical reference until further basis extrapolation or an
independent ordered-grid continuum calculation is supplied.
