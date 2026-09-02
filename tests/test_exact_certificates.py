import subprocess
import sys
from pathlib import Path


def test_tagged_cayley_exact_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "math" / "certificates" / "verify_tagged_cayley.py"),
            "--verify",
            str(root / "math" / "certificates" / "tagged_cayley_certificate.json"),
        ],
        check=True,
        cwd=root,
    )


def test_fixed_bond_cayley_exact_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(
                root
                / "math"
                / "certificates"
                / "verify_fixed_bond_cayley.py"
            ),
            "--verify",
            str(
                root
                / "math"
                / "certificates"
                / "fixed_bond_cayley_certificate.json"
            ),
        ],
        check=True,
        cwd=root,
    )


def test_rational_legendre_pointwise_exact_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "verify_rational_legendre_pointwise_reduction.py"),
        ],
        check=True,
        cwd=root,
    )


def test_triangular_pair_lc_agp_exact_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(
                root
                / "math"
                / "certificates"
                / "verify_triangular_pair_collapse.py"
            ),
            "--verify",
            str(
                root
                / "math"
                / "certificates"
                / "triangular_pair_lc_agp_certificate.json"
            ),
        ],
        check=True,
        cwd=root,
    )


def test_mat2_pair_lc_agp_exact_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(
                root
                / "math"
                / "certificates"
                / "verify_mat2_pair_collapse.py"
            ),
            "--verify",
            str(
                root
                / "math"
                / "certificates"
                / "mat2_pair_lc_agp_certificate.json"
            ),
        ],
        check=True,
        cwd=root,
    )


def test_truncated_polynomial_pair_lc_agp_exact_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(
                root
                / "math"
                / "certificates"
                / "verify_truncated_polynomial_pair_collapse.py"
            ),
            "--verify",
            str(
                root
                / "math"
                / "certificates"
                / "truncated_polynomial_pair_lc_agp_certificate.json"
            ),
        ],
        check=True,
        cwd=root,
    )


def test_alternating_word_pair_lc_agp_exact_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(
                root
                / "math"
                / "certificates"
                / "verify_alternating_word_pair_collapse.py"
            ),
            "--verify",
            str(
                root
                / "math"
                / "certificates"
                / "alternating_word_pair_lc_agp_certificate.json"
            ),
        ],
        check=True,
        cwd=root,
    )


def test_sparse_path_apg_permanent_exact_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(
                root
                / "math"
                / "certificates"
                / "verify_sparse_path_apg_permanent.py"
            ),
            "--verify",
            str(
                root
                / "math"
                / "certificates"
                / "sparse_path_apg_permanent_certificate.json"
            ),
        ],
        check=True,
        cwd=root,
    )


def test_approximate_exterior_gate_exact_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(
                root
                / "math"
                / "certificates"
                / "verify_approximate_exterior_gate.py"
            ),
            "--verify",
            str(
                root
                / "math"
                / "certificates"
                / "approximate_exterior_gate_certificate.json"
            ),
        ],
        check=True,
        cwd=root,
    )


def test_statistics_carrier_obstruction_exact_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(
                root
                / "math"
                / "certificates"
                / "verify_statistics_carrier_obstruction.py"
            ),
            "--verify",
            str(
                root
                / "math"
                / "certificates"
                / "statistics_carrier_obstruction_certificate.json"
            ),
        ],
        check=True,
        cwd=root,
    )


def test_four_form_direct_sum_exact_certificate() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "math" / "four_forms" / "verify_direct_sum_control.py"),
            "--verify",
            str(
                root
                / "math"
                / "four_forms"
                / "direct_sum_16_rank24_certificate.json"
            ),
        ],
        check=True,
        cwd=root,
    )
