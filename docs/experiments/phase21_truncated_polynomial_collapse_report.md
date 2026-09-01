# Phase 21 interim report: one-generator growing memory collapses

## Outcome

Candidate I1, the growing local algebra `C[z]/(z^d)`, is **tractable but
rejected as a beyond-LC-AGP family**. For N=2M and arbitrary coefficient-space
boundaries,

```text
K <= M(d-1)+1
```

scalar AGPs suffice exactly. The result remains polynomial even when the
radical nilpotency index `d` grows with the problem.

## Verification

The standalone exact-rational verifier covers every boundary basis functional
for all 16 `(M,d)` pairs with `1<=M,d<=4`. It compares the complete homogeneous
polynomial coefficient tables, not sampled numerical values. Certificate hash:

```text
07a222de3f44ced1b3fe155638299fdb8443a54f3d9f36479b994f32f9f0fd55
```

## Novelty decision

The construction is a project-specific application of established polynomial
interpolation, Waring decomposition, and Veronese jet/osculating geometry. It
also lands directly in the known LC-AGP family, including modern AGP-CI work
that explicitly uses border-rank-motivated deformations. No ansatz or
contraction novelty is claimed.

Exact rank and border rank are not conflated. The first jet
`M F_0^(M-1)F_1` has exact complex Waring rank `M` for `M>=2` but border rank
two. Phase 21 uses an exact `O(Md)` interpolation identity; a singular
small-parameter approximation would not meet the gate.

## Decision

Reject I1 without continuous variational experiments. The next theory target
must introduce at least two noncommuting memory branches or a controlled quiver
width while its path depth grows. It must be tested against both the Phase 13
hard tagged construction and polynomial LC-AGP reorganization before any
numerical solver work.

## Reproduction

```powershell
.\.venv\Scripts\python math\certificates\verify_truncated_polynomial_pair_collapse.py --verify math\certificates\truncated_polynomial_pair_lc_agp_certificate.json
.\.venv\Scripts\python -m pytest -q tests\test_exact_certificates.py
```
