# Fixed-state graded memory and weighted automata

## Established representation language

Finite weighted automata compute word weights through finite matrix linear
representations. The corresponding recognizable series are the rational
noncommutative formal power series; Balle--Panangaden--Precup use this relation
and the associated Hankel matrices in their canonical-form analysis
[@BallePanangadenPrecup2015WeightedAutomata]. Bell--Smertnig explicitly review
the Schuetzenberger equivalence between rational series, linear
representations, and weighted finite automata before studying unambiguous Polya
series [@BellSmertnig2021NoncommutativePolya].

The Phase 21 alternating-word embedding is therefore standard finite-state path
bookkeeping in a specialized graded algebra. No priority claim is made for
matrix representations, path automata, rational series, or coefficient
extraction.

## Project-specific negative classification

The FEMPS-specific step is to observe that physical two-forms commute, so after
evaluating a fixed number of grading counters, any fixed-width matrix power is
a homogeneous polynomial in a fixed number of scalar pair forms. Exact Waring
interpolation then makes it a polynomial-size LC-AGP.

This gives a new boundary statement only in the project's restricted sense:
fixed automaton width and fixed commuting-counter count cannot produce the
desired exactly contractible state family beyond LC-AGP. It is not asserted as
a new result about weighted automata or rational series themselves.

The general statement does not cover growing automaton width or a growing
number of independent noncommutative counters. Those regimes include both the
remaining candidate space and the Phase 13 tagged hardness construction.
