# Active execution plan: Phase 39 Combined-Manuscript Closure and Distinctiveness Gate

## Objective

Close the single combined structural/no-go manuscript before starting any
independent method-paper work. Preserve the existing diagonal-path solver as an
NOCI-equivalent numerical exercise, and determine which non-NOCI route or
matched comparator could supply a genuinely differentiating FEMPS result.

## Frozen publication scope

- The sole submission candidate is `math/femps_no_go_manuscript.tex`.
- Restore explicit Structural results I--III and the exact `chi=2` versus
  maximum-bond-three norm-hardness boundary.
- Incorporate only bounded, reproducible restricted-solver numerics needed to
  demonstrate the algorithm-design consequences of the theory.
- Do not develop or submit `docs/paper/femps_method_manuscript.tex` as a second
  paper. It is a frozen internal working note.
- Do not claim that diagonal-path FEMPS is distinct from NOCI.

## Primary work

1. Complete proof, citation, prior-art, and claim-boundary review of the
   combined manuscript, including reviewer comments on the original Theorems
   1--3, fixed-bond contraction, AGP, odd/even forms, FCI/DMRG, numerical
   wording, and AI disclosure.
2. Audit at most two routes to a non-NOCI differentiator:
   - explicit-correlation/exterior carrier structure with a plausible matched
     functional-basis `D`-convergence advantage;
   - matched Li--Waintal and same-orbital-basis DMRG comparators.
3. For each route, give the represented state class, exact/approximate
   contraction status, time and memory dependence, antisymmetry residual,
   smallest decisive benchmark, and a failure rule.
4. Select one route for a preregistered small-system experiment only after the
   audit. A clean N4,D8 NOCI-equivalent run may be retained as supporting data,
   but it cannot by itself open a method paper or satisfy the distinctiveness
   gate.

## Acceptance gates

- One combined manuscript PDF passes proof/source checks and visual review.
- The three structural results and the `chi=2` conjectural boundary are
  directly discoverable in the source and manuscript status map.
- Every diagonal-path claim is explicitly labeled NOCI-equivalent numerical
  evidence.
- At most one next algorithm route is selected, with a preregistered comparison
  capable of falsifying the claimed differentiator.
- No second method manuscript is opened.

## Gate for a future method paper

A separate paper remains closed until a non-NOCI first-quantized continuous
FEMPS demonstrates at least one of:

1. explicit correlation with a reproducible matched `D`-convergence advantage;
2. a clear accuracy, stability, memory, or complexity tradeoff against
   Li--Waintal and/or same-basis DMRG.

More Slater terms, more seeds, a larger NOCI-equivalent system, or backend
speedups do not pass this gate.

## Failure rule

If no candidate can plausibly exceed NOCI under the bounded audit, report that
the current FEMPS form is not yet an independent practical method. Keep the
combined structural paper and numerical controls, and do not manufacture a
second-paper claim through additional convergence plots.
