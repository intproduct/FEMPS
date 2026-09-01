"""Write the registered Phase 32 operator and D=12 resource preflight.

This audit must be generated before the first N=6,D=12 production run.  It
uses the completed Phase 29 blind D=10,K=4 runs as the empirical anchor and a
deliberately conservative D^2 L work model.  The estimate is a local resource
gate, not an asymptotic complexity claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import torch

from femps.exterior import diagonal_path_structural_counts
from femps.hamiltonians import soft_coulomb_operator


PARTICLES = 6
TERMS = 4
DIMENSIONS = (8, 10, 12)
QUADRATURE = 128
COUPLING = 1.0
SOFTENING = 1.0
ANCHOR_DIMENSION = 10
CPU_RSS_CAP_BYTES = 2 * 1024**3
WALL_TIME_CAP_SECONDS = 600.0
RESOURCE_MARGIN = 1.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _operator_audit(dimension: int) -> dict:
    _, diagnostics = soft_coulomb_operator(
        dimension,
        quadrature_order=QUADRATURE,
        coupling=COUPLING,
        softening=SOFTENING,
        relative_threshold=1e-13,
        factorization_backend="physical",
        dtype=torch.complex128,
        device="cpu",
    )
    counts = diagonal_path_structural_counts(
        PARTICLES, dimension, TERMS, diagnostics.retained_rank
    )
    return {
        "D": dimension,
        "Q": QUADRATURE,
        "physical_operator_svd_rank": diagnostics.retained_rank,
        "dense_relative_factorization_error": (
            diagnostics.dense_relative_factorization_error
        ),
        "exterior_ci_dimension": math.comb(dimension, PARTICLES),
        "forbidden_particle_tensor_coefficients": dimension**PARTICLES,
        "structural_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor-artifact",
        type=Path,
        default=Path(
            "docs/experiments/results/phase29_n6_multiseed_stability.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/results/phase32_n6_resource_audit.json"),
    )
    parser.add_argument("--probe-device", default="cuda:2")
    args = parser.parse_args()

    source = json.loads(args.anchor_artifact.read_text(encoding="utf-8"))
    if source["model"] != {
        "N": PARTICLES,
        "D": ANCHOR_DIMENSION,
        "K": TERMS,
        "Q": QUADRATURE,
        "coupling": COUPLING,
        "softening": SOFTENING,
    }:
        raise ValueError("Phase 29 anchor does not match the registered model")
    if not source["acceptance"]["multiseed_pass"]:
        raise ValueError("Phase 29 anchor did not pass its registered gate")

    audits = [_operator_audit(dimension) for dimension in DIMENSIONS]
    by_dimension = {point["D"]: point for point in audits}
    anchor_rank = by_dimension[ANCHOR_DIMENSION]["physical_operator_svd_rank"]
    target_rank = by_dimension[12]["physical_operator_svd_rank"]
    anchor_time = max(
        point["total_elapsed_seconds_this_call"] for point in source["points"]
    )
    anchor_rss = max(point["peak_cpu_rss_bytes"] for point in source["points"])

    # D^2 covers orbital/operator projections; L covers the factorized pair sum.
    # The explicit 1.25 margin also absorbs CI verification and allocator variance.
    work_ratio = (12 / ANCHOR_DIMENSION) ** 2 * (target_rank / anchor_rank)
    estimated_time = anchor_time * work_ratio * RESOURCE_MARGIN
    estimated_cpu_rss = math.ceil(anchor_rss * work_ratio * RESOURCE_MARGIN)
    probe_device = torch.device(args.probe_device)
    if probe_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("the registered Blackwell diagnostic requires CUDA")
    selected_properties = torch.cuda.get_device_properties(probe_device)
    gpu_probe = torch.randn(
        (TERMS, 12, PARTICLES),
        dtype=torch.complex128,
        device=probe_device,
        requires_grad=True,
    )
    probe_value = torch.linalg.det(
        gpu_probe.conj().transpose(-2, -1) @ gpu_probe
    ).real.sum()
    probe_value.backward()
    probe_peak = int(torch.cuda.max_memory_allocated(probe_device))

    admitted = bool(
        estimated_time <= WALL_TIME_CAP_SECONDS
        and estimated_cpu_rss <= CPU_RSS_CAP_BYTES
    )
    artifact = {
        "schema_version": 1,
        "experiment": "phase32_N6_D8_D10_D12_operator_and_resource_preflight",
        "evidence_level": "numerical",
        "scientific_boundary": (
            "local preflight from one D=10 empirical anchor; not an asymptotic fit"
        ),
        "model": {
            "N": PARTICLES,
            "K": TERMS,
            "dimensions": list(DIMENSIONS),
            "Q": QUADRATURE,
            "coupling": COUPLING,
            "softening": SOFTENING,
        },
        "production_contract": {
            "route": "exact determinant-transition diagonal-path FEMPS",
            "device": "cpu",
            "adam_steps": 160,
            "lbfgs_steps": 80,
            "truth_state_initialization": False,
            "enumerate_virtual_paths": False,
            "materialize_particle_tensor": False,
        },
        "backend_audit": {
            "blackwell_probe_device": args.probe_device,
            "blackwell_device_name": torch.cuda.get_device_name(probe_device),
            "blackwell_total_memory_bytes": selected_properties.total_memory,
            "complex128_autograd_probe_pass": bool(torch.isfinite(probe_value)),
            "complex128_autograd_probe_peak_bytes": probe_peak,
            "N6_D10_K4_registered_runtime_gate": {
                "status": "stopped_after_600_second_limit",
                "completed_scientific_point": False,
                "diagnosis": (
                    "Python-level transition/factor loops launch many small kernels"
                ),
            },
            "decision": (
                "use the Phase-29-validated CPU backend for registered Phase 32 "
                "production; retain Blackwell only for future vectorized kernels"
            ),
        },
        "operator_audit": audits,
        "empirical_anchor": {
            "artifact": args.anchor_artifact.as_posix(),
            "sha256": _sha256(args.anchor_artifact),
            "D": ANCHOR_DIMENSION,
            "K": TERMS,
            "runs": len(source["points"]),
            "maximum_elapsed_seconds": anchor_time,
            "maximum_peak_cpu_rss_bytes": anchor_rss,
        },
        "D12_estimate": {
            "model": "D_squared_times_physical_SVD_rank",
            "work_ratio_vs_D10": work_ratio,
            "multiplicative_margin": RESOURCE_MARGIN,
            "estimated_wall_time_seconds": estimated_time,
            "estimated_peak_cpu_rss_bytes": estimated_cpu_rss,
            "registered_wall_time_cap_seconds": WALL_TIME_CAP_SECONDS,
            "registered_peak_cpu_rss_cap_bytes": CPU_RSS_CAP_BYTES,
            "admitted_before_production": admitted,
        },
        "acceptance": {
            "factorization_pass": all(
                point["dense_relative_factorization_error"] <= 1e-11
                for point in audits
            ),
            "blackwell_probe_pass": bool(torch.isfinite(probe_value)),
            "D12_resource_gate_pass": admitted,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    _write(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "D12_resource_gate_pass": admitted,
                "estimated_wall_time_seconds": estimated_time,
                "estimated_peak_cpu_rss_bytes": estimated_cpu_rss,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
