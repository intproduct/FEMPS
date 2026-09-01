# Novelty status

The novelty audit is active and incomplete. The working name FEMPS is not a
priority claim. Every candidate claim must be checked against the works in
`references/novelty_matrix.md`, especially first-quantized MPS, HS-MPS,
Grassmann/graded tensor networks, symmetry structural/degeneracy decompositions,
and determinant-carrier correlation ansätze.

Phase 12 found a direct overlap that materially narrows the project claim:
Uemura--Kasamatsu--Sugino (2015) already use deterministic linear combinations
of independently optimized AGPs with quadratic term-count cost; Dutta et al.
(2021) and Kawasaki--Gao--Scuseria (2026) further develop nonorthogonal LC-AGP
and AGP-CI constructions. The present finite-AGP implementation is therefore a
validated fallback and benchmark baseline, not a new ansatz class. The two
tested beyond-LC-AGP claims---generic matrix-wedge exact contraction and a
universal statistics-carrier/correlation-multiplicity tensor product---are now
closed negatively. The former independent four-form program is parked after
its exact seven- and eight-dimensional checkpoints. The active question is now
whether a strictly scoped FEMPS solver can pass controlled continuous-physics
benchmarks under those no-go constraints.

Phase 13 first ruled out the generic exact branch at growing bond by an
explicit tagged Cayley-determinant reduction; the direct construction plus the
structured CHSS boundary strengthens this to fixed maximum bond two. Thus “generic matrix-wedge FEMPS with polynomial exact contraction”
is no longer an admissible novelty claim. A future method claim must instead
identify a systematically improvable restricted algebra with a surviving
novelty boundary, use a controlled approximate estimator, or adopt an
explicitly named alternative first-quantized route. Phase 28 initially chooses
the nonbranching diagonal-path FEMPS, exactly a sum of `K` nonorthogonal Slater
determinants. That is a valid restricted FEMPS algorithm target but is close to
nonorthogonal selected CI and is not claimed as a new ansatz. The one-body quantity
`N/Tr(gamma^2)` is retained only as a correlation diagnostic, not a novel
entanglement measure or a canonical FEMPS bond spectrum.

Ordered-sector/interparticle-distance functional TN remains a control and is
not called FEMPS. Li--Waintal 2026 already establishes
the first-quantized ordered-distance MPS direction; any surviving contribution
must be stated narrowly as the 2201 continuous orthonormal functional-basis
operator/AD integration, its controlled continuum benchmarks, and the new
matrix-wedge/ordinary-particle-TT no-go theory. No priority claim is attached
to ordered coordinates or distance-variable MPS.

ADR 0028 withdraws the later two-paper split. The diagonal-path solver is
finite NOCI and now appears only as a numerical section of the single combined
structural/no-go manuscript. Phase 39 selects a symmetric explicit correlator
times an exterior carrier only as a candidate experiment: Jastrow factors,
backflow, determinant carriers, tensorized backflow, and VMC are prior art, so
the ansatz form alone is not a novelty claim. Phase 40 must demonstrate an
independently reproduced `D`-convergence or matched-cost advantage beyond
optimized fixed-`K` NOCI before any new publication decision. No second-paper
drafting occurs while that gate is closed.

Phase 15 closes only the finite-grid contraction gate: the exact hard-charge
gap MPS, `O(N^2(L-N))` raw operator bond, and native AD optimizer are an
implementation/evidence result, not a new ansatz claim. The subsequent Phase 16
gate asked whether full-line center-of-mass and Dirichlet half-line distance bases can
retain the distinctive 2201 orthonormal functional-operator calculus with
controlled continuum convergence. Until that comparison is complete, the
ordered-distance branch has no affirmative method-priority claim.

Phase 16 closes Gate D at controlled small-system scope. Implementation-level
review confirms that the result is a bridge between, rather than a replacement
for, its two direct parents: Hong et al.'s orthonormal continuous
functional-operator/AD construction and Li--Waintal's ordered-distance
first-quantized MPS. The surviving package is the exact COM/gap calculus,
Dirichlet and unbounded half-line bases, continuum mixed-derivative and
soft-Coulomb MPOs, native global AD, and an independently separated
`D/scale/K/chi/optimization` error budget. This is a reproducible integration
and evidence contribution. It is not an affirmative ansatz-priority claim,
and it is not called FEMPS.

Phase 17 closes Gate E at controlled N=6 scope without broadening that claim.
The implemented Fourier--Bessel/odd-Hermite interaction, compact four-real-state
all-pair recurrence, and global compression audits are an integration and
evidence result. The recurrence removes direct-pair growth at fixed Fourier
order, but no priority claim is made for ordered coordinates, first-quantized
distance MPS, or scalable fermionic tensor networks. Hong et al. and
Li--Waintal remain the direct method parents. The N=6 basis-dominated error and
temporary dense raw-MPO storage preclude an asymptotic accuracy/resource claim.

Phase 18 closes the core Gate F criteria at one controlled N=8 point, again
without broadening the novelty claim. The Lowdin-orthonormalized two-scale
half-line basis and incremental sparse-recurrence MPO builder are basis/
implementation choices inside the established ordered first-quantized and
functional-basis parent methods. They reduce measured error and construction
memory but do not establish a new ansatz class. The retained raw-gradient
auxiliary miss, numerical rather than continuum N=8 reference, and chi-32
local-solver resource rejection explicitly preclude an asymptotic or method-
priority claim.

Phase 19 closes those two operational Gate F qualifications without broadening
the novelty claim. A bounded local effective-Hamiltonian contraction order, a
left-gauge physical-tangent audit, matched MPO-bond training, and a blind D12
refinement are implementation and evidence controls inside the same Hong et
al./Li--Waintal parent route. The exterior D14 value remains a numerical
reference, the N=2/4/6/8 trend does not admit N=10, and the route is still not
called FEMPS. Phase 20 therefore returns to the unresolved novelty boundary: a
restricted exterior correlation structure that is both genuinely beyond
finite LC-AGP/Gaussian families and exactly polynomially contractible, or a
documented negative classification.

Phase 20 supplies a broader negative classification. The noncommutative `2 x 2`
upper-triangular candidate first collapses to at most `binom(M+2,2)+2` AGPs.
More generally, if a finite-dimensional complex coefficient algebra has both
uniformly bounded semisimple matrix-block size and uniformly bounded radical
nilpotency index, every arbitrary-boundary pair power has a polynomial-size
exact LC-AGP expansion. The fully noncommutative semisimple `Mat_2` base case
already needs at most `binom(M+3,3)` terms. Thus neither noncommutativity nor
bounded radical memory creates a new exterior contraction class; Gate H closes
negatively for this candidate class. The necessary escape boundary is a
growing simple block or growing radical depth with additional structure, or an
explicitly approximate contraction. This is not an affirmative method claim.

Phase 21 closes the weakest growing-memory escape negatively. For the local
algebra `C[z]/(z^d)`, arbitrary-boundary pair powers admit an exact LC-AGP
expansion with at most `M(d-1)+1` terms even when `d` grows. This is a
coefficient-extraction/Veronese-jet identity squarely inside established Waring,
osculating, and nonorthogonal AGP territory. Exact and border rank are kept
separate: the first jet has exact rank M but border rank two. The result narrows
the next candidate to genuinely multibranch noncommutative growing memory; it
does not create a new method claim.

The second Phase 21 candidate adds genuine noncommutativity and two alternating
branches with dimension `2d-1`, but embeds in `Mat_2(C[z]/z^d)` and collapses to
at most `[M(d-1)+1] binom(M+3,3)` AGPs. More generally, fixed matrix-state width
over a fixed number of commuting grading counters always gives a polynomial
LC-AGP expansion. Finite-state matrix representations are established weighted-
automata/rational-series machinery. Gate I therefore closes negatively for
fixed-state graded memory; the next exact boundary requires growing state width
or growing independent noncommutative counters and remains subject to the Phase
13 hardness obstruction.

Phase 22 closes the weakest growing-width boundary negatively. An upper-
bidiagonal endpoint pair matrix has a unique virtual path, but its state is
exactly the established antisymmetrized product of geminals (APG). APG-to-AGP
Fischer decompositions and permanent-valued APG/APIG determinant coefficients
are direct prior art. In a paired-orbital specialization, the path coefficient
is an arbitrary 0--1 permanent and its normalized squared norm is
`perm(A)^2/(M!)^2`; hence even bandwidth one is generically #P-hard to contract
exactly. This is a project-specific gate embedding, not a claim to have
introduced APG, its Waring decomposition, or its permanent structure. Ordinary
polynomial Waring rank is also not asserted as a physical LC-AGP lower bound
after exterior quotienting. Phase 23 therefore consolidates a no-go theorem
package rather than opening another exact solver branch.

Phase 23 makes that package manuscript-explicit. The shortest generic exact
contraction obstruction is the bandwidth-one APG permanent, while the earlier
tagged Cayley theorem remains an independent classification of growing
noncommutative order memory. Fixed `Mat_2` pair powers are on the polynomial
LC-AGP side, correcting a stale Phase 14 analogy. No universal no-go or
approximation-hardness claim is made. Any approximate exterior method now
requires a separate error-certified Gate K and a fresh audit against established
APG selection, low-rank, and stochastic contraction methods.

Phase 24 closes generic relative approximation negatively without making the
invalid inference from exact hardness. The Jerrum--Sinclair--Vigoda FPRAS for
entrywise-nonnegative matrices and Gurvits-type additive estimators remain
prior-art positive boundaries. On the different, admitted class of real-PSD
coefficient arrays, however, the sparse APG identity transfers Meiburg's
relative inapproximability directly to the squared norm: a generic PRAS would
imply `RP=NP`. The project-specific content is this FEMPS/APG transfer and the
simultaneous Rayleigh denominator certificate, not any permanent algorithm or
hardness theorem. Gate K therefore closes without a solver. The question passed
to Phase 25 was whether a canonical statistics-carrier/correlation-multiplicity
factorization can satisfy Slater multiplicity one, safe truncation, and
polynomial contraction without hiding the same permanent or binomial cost.

Phase 25 answers that question negatively for the direct universal tensor
product. A two-Slater family has stable one-cut rank `N+2`, whereas Slater
multiplicity one forces a fixed carrier dimension `N`; the required
divisibility fails for every `N>=3`. This does not challenge established
symmetry-adapted TN: particle permutation antisymmetry has a one-dimensional
sign irrep, and Hamiltonian-specific charge/multiplet decompositions remain
valid. Nor is state-adaptive Slater/secant geometry new. The project-specific
result is the exact dimension obstruction and its link to compact-input
permanent contraction. “Canonical FEMPS correlation spectrum” is therefore no
longer an admissible claim for the current construction. Phase 26 shifts from
method invention to a manuscript-level proof and novelty audit.

Phase 26 resolves the reviewer-requested fixed-small-bond checkpoint more
strongly than the earlier tagged construction. Site-indexed one-form cores of
bond two preserve the row order of a `Mat_2` Cayley determinant directly. A
subsequent CHSS audit uses its structured `a I_2+b J_2` output to extract the
nonnegative `a+b=4^(3m)#SAT` coefficient with fixed boundaries, proving the
exact squared-norm obstruction already at bond two. A bond-one scalar reference
raises the maximum bond to three only when a general signed output must be
recovered by polarization. Conditional on the
published noncommutative-determinant hardness theorem, generic exact
one-form-FEMPS contraction is therefore already #P-hard at fixed `chi<=3`.
This is a project-specific representation transfer, not a new complexity
theorem. It does not conflict with the LC-AGP collapse of fixed `Mat_2`
homogeneous pair powers, which symmetrize factor order. The defensible Phase 26
output is now a unified no-go/classification manuscript plus exact certificates;
it is not an affirmative scalable FEMPS method claim.
