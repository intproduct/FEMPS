# Completed execution plan: Phase 43 controlled N>2 explicit-correlation backend

## Closure

Phase 43 closed on 2026-09-02 with the ADR-0032 fixed-state estimator gate
passing. Two frozen `N=2` chains agree with deterministic quadrature and
gradient truth under the registered absolute/statistical tolerances. The
`N=4` noninteracting Slater has exact energy 8, numerical variance
`4.96e-31`, and bitwise-identical forced-resume and clean samples. All sampled
antisymmetry residuals are below `7.2e-16`.

The implementation, raw coordinates, artifact verifier, report, and full test
result are committed at `e23edbf`. This closes estimator validation only. No
interacting `N=4` result, external replication, scalable-solver claim, or
Paper B was admitted.

ADR 0033 and Phase 44 now separately freeze the interacting production gate.

## Frozen scientific boundary

Phase 40 established only a bounded `N=2` low-`D` differentiator. Phase 43
established that its coordinate estimator and resume machinery generalize to
small `N`, including the exact `N=4` noninteracting limit. The near-stationary
`N=2` fixture makes the gradient comparison an absolute-error implementation
check, not strong relative-gradient evidence away from stationarity.

Any practical-method claim still requires interacting optimization, systematic
`D` behavior, clean and external reproduction, and ultimately a matched
Li--Waintal or same-basis-DMRG comparison.
