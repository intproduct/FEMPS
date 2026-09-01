"""Post-selection-only comparator loader for the Phase 44 firewall."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


TRUTH_SOURCE = Path(
    "docs/experiments/results/soft_coulomb_n4_truth_sweep.json"
)
PHASE37_SOURCE = Path(
    "docs/experiments/results/phase37_slater_source_solver.json"
)
D8_NOCI_SOURCE = Path(
    "docs/experiments/results/soft_coulomb_n4_k_hierarchy.json"
)
REFERENCE_ENERGY = 11.023082853674637
EXPECTED_CI = {
    4: 11.085944151108343,
    6: 11.023837713203346,
    8: 11.023278984749750,
}
EXPECTED_NOCI = {
    4: 11.085944151108343,
    6: 11.023837713691630,
    8: 11.023284391447700,
}


def _normalized_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_frozen_comparators() -> dict[str, Any]:
    """Open and validate comparators only after the selection-ledger boundary."""

    truth = json.loads(TRUTH_SOURCE.read_text(encoding="utf-8"))
    phase37 = json.loads(PHASE37_SOURCE.read_text(encoding="utf-8"))
    d8_noci = json.loads(D8_NOCI_SOURCE.read_text(encoding="utf-8"))
    ci = {point["D"]: point["ground_energy"] for point in truth["basis_scan"]}
    reference = ci[14]
    d6_k4 = next(
        stage["optimizer_result"]["energy"]
        for stage in phase37["clean_result"]["stages"]
        if stage["terms"] == 4
    )
    d8_k4 = next(
        level["joint"]["final_energy"]
        for level in d8_noci["levels"]
        if level["K"] == 4
    )
    noci = {4: ci[4], 6: d6_k4, 8: d8_k4}
    if not math.isclose(reference, REFERENCE_ENERGY, rel_tol=0.0, abs_tol=1e-14):
        raise AssertionError("D14 numerical reference changed")
    for basis_order in (4, 6, 8):
        if not math.isclose(
            ci[basis_order], EXPECTED_CI[basis_order], rel_tol=0.0, abs_tol=1e-14
        ):
            raise AssertionError(f"D{basis_order} CI comparator changed")
        if not math.isclose(
            noci[basis_order],
            EXPECTED_NOCI[basis_order],
            rel_tol=0.0,
            abs_tol=5e-13,
        ):
            raise AssertionError(f"D{basis_order} NOCI comparator changed")
    return {
        "d14_numerical_reference": reference,
        "d14_is_not_a_continuum_bound": True,
        "ci": {str(key): value for key, value in ci.items() if key in (4, 6, 8)},
        "noci_k4": {str(key): value for key, value in noci.items()},
        "noci_absolute_reference_errors": {
            str(key): abs(value - reference) for key, value in noci.items()
        },
        "source_hashes": {
            path.as_posix(): _normalized_sha256(path)
            for path in (TRUTH_SOURCE, PHASE37_SOURCE, D8_NOCI_SOURCE)
        },
    }


__all__ = ["load_frozen_comparators"]
