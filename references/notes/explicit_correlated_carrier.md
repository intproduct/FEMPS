# Explicit-correlator exterior carrier: prior-art and feasibility audit

## Candidate state

The candidate retained by Phase 39 is

```text
Psi_(theta,A)(x_1,...,x_N)
  = exp(sum_(i<j) u_theta(x_i,x_j)) Psi_A^exterior(x_1,...,x_N),
```

where `u_theta(x,y)=u_theta(y,x)` and `Psi_A^exterior` is a first-quantized
continuous functional-basis exterior carrier. For every particle permutation
`P`, the correlator is invariant and the carrier changes by `sgn(P)`, so the
product remains exactly antisymmetric. Every numerical calculation must still
report a swap/materialization residual.

The first prototype uses a `chi=1` Slater carrier and a Gaussian pair basis for
`u_theta`. This is a carrier--correlation extension of FEMPS, not a claim that
the multiplied state remains inside the finite Galerkin space `Lambda^N V_D`.
That distinction is the source of a possible explicit-correlation basis
advantage and must be stated in every comparison.

## Nearest prior art

- Real-space Slater--Jastrow, Pfaffian, and backflow wave functions are
  established QMC ansatzes; the carrier or symmetric multiplier is not new
  [@BajdichMitasWagnerSchmidt2008PfaffianQMC].
- Zhou--Zhou--Liang use tensor representations of backflow corrections for
  lattice/Fock-space models and molecular minimal bases
  [@ZhouZhouLiang2024TensorBackflow].
- Bortone--Rath--Booth give a systematically improvable CP decomposition of
  backflow in a fixed-basis, second-quantized VMC formulation and compare it
  with DMRG [@BortoneRathBooth2025CPDBackflow]. It cannot be relabeled FEMPS.
- FermiNet and PauliNet already provide much more expressive first-quantized
  real-space determinant/Jastrow/backflow solvers. The surviving project
  question is therefore not first-quantized priority or a new Jastrow state.
- Li--Waintal own the ordered-coordinate first-quantized MPS direction. The
  project's ordered COM/gap implementation remains an attributed comparator,
  not the exterior-correlator method.

The only potentially distinct package is the integration of an exterior
functional carrier, explicit independent carrier/correlator basis controls,
and an error-audited deterministic/VMC solver. A method claim requires measured
`D`-convergence or complexity evidence; the representation alone is prior art.

## Complexity audit

For the bounded `N=2` deterministic oracle with quadrature order `Q`, carrier
basis order `D`, and `P` pair features:

```text
time   = O(Q^2 (D + P)),
memory = O(Q^2 + D Q),
```

and the `Q^2` grid is explicitly materialized. This is only a truth/AD oracle.

For a future single-Slater carrier VMC backend, one coordinate sample requires
roughly

```text
pair correlator and derivatives: O(P N^2),
orbital values/derivatives:      O(D N^2),
determinant/local updates:       O(N^3),
memory:                          O(N^2 + D N + P N).
```

These are evaluation costs, not an end-to-end mixing or variance theorem.
Autocorrelation time, local-energy variance, optimizer stability, and failure
probability must be measured. A general branching matrix-wedge carrier remains
subject to the exact-contraction obstruction; the pilot does not bypass it.

## Route comparison

| Route | Scientific value | Blocking issue | Decision |
|---|---|---|---|
| Symmetric explicit correlator times exterior carrier | Non-NOCI compact correlation is possible; exact antisymmetry is structural; a direct `D`-convergence test exists | General contraction becomes stochastic; ansatz overlaps established Jastrow/backflow literature | Select as the one main candidate, with novelty limited to functional/exterior integration and measured tradeoffs |
| Li--Waintal / same-basis DMRG implementation | Supplies mandatory accuracy, entanglement, memory, and timing controls | Ordered coordinates and DMRG are existing methods, not FEMPS; no new carrier results | Retain as comparator work, not a second candidate method |

No separate backup ansatz is activated. If the explicit-correlator gate fails,
the honest result is that the present FEMPS program lacks a distinct practical
solver; more NOCI terms or ordered-coordinate calculations do not repair it.

## Small-system materialization evidence

The exploratory `N=2` prototype uses a canonical `D=2` Slater carrier embedded
in a four-function implementation buffer and five Gaussian pair features. It
is evaluated by deterministic product Gauss--Hermite quadrature and projected
back into growing harmonic spaces only as a truth audit.

The projection has antisymmetric matrix ranks `4,6,8,10,12` at projection
orders `D=4,6,8,10,12`, respectively, under a relative `1e-10` threshold. Thus
one correlated carrier requires Slater ranks `2,3,4,5,6` in those bounded
projections, whereas the uncorrelated carrier has Slater rank one. This is
numerical evidence, not a theorem for arbitrary `D` or correlator parameters.

The optimized exploratory energy is `2.553833129442`, compared with
`2.564890847246` for the same uncorrelated carrier and
`2.553828651107` for the independent relative-coordinate grid reference. The
absolute reference error is `4.478e-6`, the energy variance is `2.136e-5`, the
Q128-to-Q160 norm change is `5.503e-11`, the AD/finite-difference gradient
difference is `1.509e-11`, and both correlator-symmetry and antisymmetry
residuals are zero. These values are exploratory numerical evidence and do not
pass the future method-paper gate.
