# Sparse path states, APG, and permanent structure

## Direct identification

For pair number `M`, an `(M+1) x (M+1)` upper-bidiagonal matrix of physical
two-forms with entries `Omega_(i-1,i)=F_i` and endpoint boundaries has a unique
length-`M` virtual path. Its endpoint matrix-power coefficient is

```text
(Omega^M)_(0,M) = F_1 wedge ... wedge F_M.
```

This is exactly an antisymmetrized product of geminals (APG), up to the
project's conventional factor `1/M!`. It is not a new geminal ansatz.
Johnson et al. treat products of nonorthogonal geminals and relate special
cases to APSG, GVB-PP, and AGP
[@JohnsonAyersDeBaerdemackerEtAl2022BivariationalAPG].

Kawasaki--Nakatani apply the classical Fischer/Waring identity directly to an
APG and obtain a finite sum of AGPs. Their exact formula has `2^(M-1)` terms;
they then investigate low-rank approximations because that exact term count is
exponential [@KawasakiNakatani2024LowRankAPG]. This is an exact upper bound,
not a proof of minimal AGP-CI length after quotienting by exterior relations.
In particular, ordinary polynomial Waring rank cannot be imported unchanged
through `Sym^M(Lambda^2 V) -> Lambda^(2M) V`.

## Permanent structure is established

Richer--Kim--Ayers write APG determinant coefficients as sums of permanents.
For APIG only one pairing scheme remains, but evaluating its coefficient still
requires a permanent for unrestricted geminal coefficients
[@RicherKimAyers2025GraphicalGeminals]. Moisset--Fecteau--Johnson likewise
classify generic seniority-zero geminal scalar products and density matrices as
intractable while identifying structured tractable cases such as AGP and
strongly orthogonal products [@MoissetFecteauJohnson2022GeminalRDM].

The Phase 22 reduction is the paired-orbital/APIG specialization. With
`P_j=e_(2j-1) wedge e_(2j)` and

```text
F_i = sum_j A_(i,j) P_j,
```

the top-form coefficient of `F_1 wedge ... wedge F_M` is `perm(A)`. This is a
transparent re-expression of known APG permanent structure. The project-
specific contribution is only its use as a sparse growing-width matrix-pair
Gate J counterexample, together with an exact certificate in the FEMPS
normalization convention. No priority claim is made for APG, APIG, the Fischer
identity, or the permanent formula.

Valiant proves that computing the permanent of a 0--1 matrix is complete for
the corresponding counting class [@Valiant1979Permanent]. Therefore the exact
FEMPS norm consequence is conditional only in the standard complexity-theory
sense: a polynomial exact contraction would imply a polynomial algorithm for
this #P-complete problem.

## General fixed-bandwidth walks

A tridiagonal or fixed-bandwidth endpoint coefficient is a sum over virtual
walks, each carrying an ordered product of two-forms. Since two-forms commute,
each walk term is an APG product; the whole state is a structured sum of APGs.
This arithmetic-branching-program description gives a compact formal
polynomial and a cheap recurrence only before physical exterior coefficient or
norm extraction. It does not remove the permanent obstruction, because the
upper-bidiagonal unique-path subclass is already hard.
