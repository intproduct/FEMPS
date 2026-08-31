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
