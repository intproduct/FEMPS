"""Exact diagnostics for the controlled approximate exterior gate.

The verifier uses only integers and :class:`fractions.Fraction`.  It checks
the permanent-to-APG squared-norm identity on positive, cancelling,
precision-ill-conditioned, and signed real-PSD examples.  It also checks the
a-posteriori Rayleigh-quotient bound used by Gate K.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path


Matrix = list[list[Fraction]]


def _fraction(value: int | str) -> Fraction:
    return Fraction(value)


def _permanent(matrix: Matrix) -> Fraction:
    order = len(matrix)
    return sum(
        (
            math.prod(
                matrix[row][column]
                for row, column in enumerate(permutation)
            )
            for permutation in itertools.permutations(range(order))
        ),
        start=Fraction(0),
    )


def _display(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _matrix_display(matrix: Matrix) -> list[list[str]]:
    return [[_display(value) for value in row] for row in matrix]


def _matrix_case(
    label: str,
    matrix: Matrix,
    *,
    promise: str,
    expected_permanent: Fraction,
) -> dict[str, object]:
    permanent = _permanent(matrix)
    if permanent != expected_permanent:
        raise AssertionError(f"permanent mismatch for {label}")
    order = len(matrix)
    norm_squared = permanent * permanent / math.factorial(order) ** 2
    return {
        "label": label,
        "promise": promise,
        "matrix": _matrix_display(matrix),
        "permanent": _display(permanent),
        "normalized_exterior_norm_squared": _display(norm_squared),
    }


def _energy_case(
    label: str,
    *,
    norm: Fraction,
    numerator: Fraction,
    norm_estimate: Fraction,
    numerator_estimate: Fraction,
    norm_radius: Fraction,
    numerator_radius: Fraction,
) -> dict[str, str]:
    if abs(norm - norm_estimate) > norm_radius:
        raise AssertionError(f"invalid norm radius for {label}")
    if abs(numerator - numerator_estimate) > numerator_radius:
        raise AssertionError(f"invalid numerator radius for {label}")
    if norm_estimate <= norm_radius:
        raise AssertionError(f"uncertified denominator for {label}")
    energy = numerator / norm
    estimate = numerator_estimate / norm_estimate
    error = abs(energy - estimate)
    bound = (
        numerator_radius + abs(estimate) * norm_radius
    ) / (norm_estimate - norm_radius)
    if error > bound:
        raise AssertionError(f"energy bound failed for {label}")
    return {
        "label": label,
        "norm": _display(norm),
        "numerator": _display(numerator),
        "norm_estimate": _display(norm_estimate),
        "numerator_estimate": _display(numerator_estimate),
        "norm_radius": _display(norm_radius),
        "numerator_radius": _display(numerator_radius),
        "energy": _display(energy),
        "energy_estimate": _display(estimate),
        "absolute_error": _display(error),
        "certified_bound": _display(bound),
    }


def build_certificate() -> dict[str, object]:
    positive = [
        [_fraction(1), _fraction(2)],
        [_fraction(3), _fraction(4)],
    ]
    cancelling = [
        [_fraction(1), _fraction(1)],
        [_fraction(1), _fraction(-1)],
    ]
    signed_psd = [
        [_fraction(1), _fraction("-1/2")],
        [_fraction("-1/2"), _fraction(1)],
    ]
    # Sylvester's criterion in dimension two certifies this matrix as positive
    # definite: its leading principal minors are 1 and 3/4.
    if signed_psd[0][0] <= 0:
        raise AssertionError("signed PSD leading minor failed")
    signed_psd_determinant = (
        signed_psd[0][0] * signed_psd[1][1]
        - signed_psd[0][1] * signed_psd[1][0]
    )
    if signed_psd_determinant <= 0:
        raise AssertionError("signed PSD determinant failed")

    matrix_cases = [
        _matrix_case(
            "entrywise_nonnegative",
            positive,
            promise="entrywise nonnegative",
            expected_permanent=_fraction(10),
        ),
        _matrix_case(
            "exact_cancellation",
            cancelling,
            promise="signed",
            expected_permanent=_fraction(0),
        ),
        _matrix_case(
            "signed_real_psd",
            signed_psd,
            promise="real positive definite with a negative off-diagonal",
            expected_permanent=_fraction("5/4"),
        ),
    ]

    precision_cases = []
    for bits in (4, 8, 16, 32):
        tau = Fraction(1, 1 << bits)
        matrix = [
            [_fraction(1), _fraction(1)],
            [_fraction(1), -_fraction(1) + tau],
        ]
        permanent = _permanent(matrix)
        if permanent != tau:
            raise AssertionError(f"cancellation identity failed at L={bits}")
        norm_squared = permanent * permanent / 4
        precision_cases.append(
            {
                "input_precision_bits": bits,
                "tau": _display(tau),
                "permanent": _display(permanent),
                "normalized_exterior_norm_squared": _display(norm_squared),
                "necessary_additive_permanent_tolerance_below": _display(tau),
            }
        )

    energy_cases = [
        _energy_case(
            "positive_energy",
            norm=_fraction(3),
            numerator=_fraction(5),
            norm_estimate=_fraction("31/10"),
            numerator_estimate=_fraction("49/10"),
            norm_radius=_fraction("1/5"),
            numerator_radius=_fraction("1/5"),
        ),
        _energy_case(
            "negative_energy",
            norm=_fraction(2),
            numerator=_fraction(-7),
            norm_estimate=_fraction("19/10"),
            numerator_estimate=_fraction("-36/5"),
            norm_radius=_fraction("1/5"),
            numerator_radius=_fraction("3/10"),
        ),
        _energy_case(
            "numerator_interval_crosses_zero",
            norm=_fraction(1),
            numerator=_fraction("1/20"),
            norm_estimate=_fraction("11/10"),
            numerator_estimate=_fraction(0),
            norm_radius=_fraction("1/5"),
            numerator_radius=_fraction("1/10"),
        ),
    ]

    body: dict[str, object] = {
        "schema": "femps.approximate-exterior-gate.v1",
        "arithmetic": "exact rational",
        "state_convention": "Psi_A=perm(A) P_1...P_M/M!",
        "matrix_cases": matrix_cases,
        "signed_psd_principal_minors": [
            _display(signed_psd[0][0]),
            _display(signed_psd_determinant),
        ],
        "precision_cases": precision_cases,
        "energy_bound": (
            "|E-E_tilde| <= (Delta_h+|E_tilde|Delta_n)/"
            "(n_tilde-Delta_n)"
        ),
        "energy_cases": energy_cases,
    }
    body["certificate_sha256"] = _digest(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    arguments = parser.parse_args()
    observed = build_certificate()
    if arguments.verify is not None:
        expected = json.loads(arguments.verify.read_text(encoding="utf-8"))
        if observed != expected:
            raise SystemExit("certificate mismatch")
        print(f"verified {arguments.verify} ({observed['certificate_sha256']})")
        return
    print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
