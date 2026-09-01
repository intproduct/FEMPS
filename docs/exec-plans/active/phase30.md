# Active execution plan: Phase 30 FEMPS Method Consolidation

## Objective

Turn the validated restricted diagonal-path FEMPS route into a compact,
reproducible method package suitable for external scientific evaluation. No new
particle-number expansion is part of this phase.

## Admitted evidence

- first-quantized continuous functional-basis definition and exact exterior
  antisymmetry;
- versioned solver/checkpoint contract and deterministic reproduction lineage;
- N2 analytic/reference checks;
- N4 noninteracting and interacting `D`/`K` convergence, multiseed stability,
  operator, variance, norm, symmetry, time, memory, CI/Slater/AGP/particle-TT
  comparisons;
- N6 single-lineage and three-blind-seed soft-Coulomb results with direct CI
  and bounded million-coefficient materialization;
- fixed `D=10,K=4,L=19` N4-to-N6 matched kernel-cost audit.

## Active deliverables

1. Create one machine-readable reproduction manifest listing every admitted
   method claim, command, artifact, verifier, seed, tolerance, and evidence
   level.
2. Add an independent manifest verifier that checks artifact hashes, schema
   versions, scientific labels, command targets, and all registered verifier
   results without rerunning long optimizations.
3. Produce a concise method-claim/evidence matrix separating proved structure,
   numerical evidence, limitations, and unadmitted claims.
4. Prepare paper-ready tables/figures only from committed artifacts; exploratory
   floating-point data remain explicitly numerical.
5. Decide in an ADR whether an optional second-quantized DMRG control adds
   information beyond current exact CI. It must remain an external comparator
   and may not be called FEMPS.

## Scientific and scope limits

- No N8 diagonal-path run, new high-dimensional rank search, or generic exact
  contraction claim.
- Direct CI being faster in the current truth region must be stated.
- Kernel timings at N4/N6 are two-point measurements, not asymptotic scaling.
- The practical FEMPS claim is restricted to the verified nonbranching
  multideterminant subclass.
- Any method-paper claim must link to a committed artifact and independent
  verifier in the reproduction manifest.
