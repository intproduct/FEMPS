# Phase 20 interim report: T2 candidate collapses to polynomial LC-AGP

## Outcome

Candidate H1, virtual `2 x 2` upper-triangular matrix-valued pair powers, is
**tractable but rejected as a beyond-LC-AGP candidate**. Generic coefficient
matrices in this algebra need not commute, yet every N=2M state with arbitrary
boundaries has an exact LC-AGP expansion of length at most
`binom(M+2,2)+2`.

This is a useful negative classification: simultaneous diagonalizability is
not required for polynomial LC-AGP collapse. Square-zero radical mixing only
produces a bounded family of derivative/divided-difference-like geminal terms.

## Verification

- Exact rational certificate: M=1 through 6, hash
  `f671c2c10376c39cfb8c223edafba370570b9c417e8b12bcaaeb2f0f66cf078c`.
- Complex128 exterior recurrence versus LC-AGP: M=1,2,3.
- Explicitly noncommuting coefficient sample: pass.
- Reverse-mode gradients on the allowed upper-triangular/skew tangent space:
  pass.
- Invalid lower-block coefficient rejection: pass.

The exact verifier is independent of the numerical implementation and imports
neither PyTorch nor `femps`.

## Decision

Do not benchmark H1 as a new solver family: its polynomial contraction is
already supplied by the existing finite LC-AGP engine, whose prior-art boundary
is established. H1 is the first exact base case of the subsequently proved
bounded Wedderburn--radical theorem draft. Gate H is closed negatively for that
broader candidate class in `phase20_bounded_wedderburn_report.md` and ADR 0010.

## Reproduction

```powershell
.\.venv\Scripts\python math\certificates\verify_triangular_pair_collapse.py --verify math\certificates\triangular_pair_lc_agp_certificate.json
.\.venv\Scripts\python -m pytest -q tests\test_triangular_pair.py tests\test_exact_certificates.py
```
