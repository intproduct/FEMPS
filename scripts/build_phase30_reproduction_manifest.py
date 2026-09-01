"""Build the deterministic Phase 30 FEMPS method reproduction manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from femps.algorithms import (
    ADAPTIVE_DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION,
    ADAPTIVE_DIAGONAL_PATH_RESULT_SCHEMA_VERSION,
    DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION,
    DIAGONAL_PATH_RESULT_SCHEMA_VERSION,
)


ENTRIES = (
    {
        "id": "n2_harmonic_ladder",
        "claim": "N2 harmonic E1/E2 and noninteracting N4 E3 pass exact invariants and registered D/K trends.",
        "artifact": "docs/experiments/results/phase28_diagonal_path_ladder.json",
        "verifier_module": "scripts.verify_phase28_diagonal_path_ladder",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/reproduce_phase28_diagonal_path_ladder.py",
        "verify_command": "python scripts/verify_phase28_diagonal_path_ladder.py",
        "seeds": [17],
        "registered_tolerances": {
            "norm_error": 1e-10,
            "antisymmetry_residual": 1e-12,
            "polynomial_materialized_energy_difference": 1e-10,
        },
    },
    {
        "id": "n4_harmonic_e4_closure",
        "claim": "N4 interacting harmonic diagonal-path FEMPS passes blind stability, D/K convergence, and exact comparator gates.",
        "artifact": "docs/experiments/results/phase28_e4_closure.json",
        "verifier_module": "scripts.verify_phase28_e4_closure",
        "verifier_argument": "json_payload",
        "benchmark_command": "python scripts/benchmark_phase28_e4_closure.py",
        "verify_command": "python scripts/verify_phase28_e4_closure.py",
        "seeds": [17, 23, 41],
        "registered_tolerances": {
            "finite_basis_error": 1e-3,
            "variance": 1e-2,
            "antisymmetry_residual": 1e-12,
        },
    },
    {
        "id": "n4_soft_coulomb_transferability",
        "claim": "N4 soft-Coulomb FEMPS passes three-seed D6/D8 transferability and independent K/D axes.",
        "artifact": "docs/experiments/results/phase28_soft_coulomb_transferability.json",
        "verifier_module": "scripts.verify_phase28_soft_coulomb_transferability",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/benchmark_phase28_soft_coulomb_transferability.py",
        "verify_command": "python scripts/verify_phase28_soft_coulomb_transferability.py",
        "seeds": [17, 23, 41],
    },
    {
        "id": "n4_soft_coulomb_basis_extension",
        "claim": "At fixed K4, the N4 soft-Coulomb D8-to-D12 lineage passes direct-CI and basis-convergence gates.",
        "artifact": "docs/experiments/results/phase28_soft_coulomb_basis_extension.json",
        "verifier_module": "scripts.verify_phase28_soft_coulomb_basis_extension",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/benchmark_phase28_soft_coulomb_basis_extension.py",
        "verify_command": "python scripts/verify_phase28_soft_coulomb_basis_extension.py",
        "seeds": [17],
    },
    {
        "id": "n4_soft_coulomb_high_basis_correlation",
        "claim": "At N4,D12, blind K4-to-K5 growth materially reduces same-basis CI error and variance.",
        "artifact": "docs/experiments/results/phase28_soft_coulomb_high_basis_correlation.json",
        "verifier_module": "scripts.verify_phase28_soft_coulomb_high_basis_correlation",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/benchmark_phase28_soft_coulomb_high_basis_correlation.py",
        "verify_command": "python scripts/verify_phase28_soft_coulomb_high_basis_correlation.py",
        "seeds": [2812],
    },
    {
        "id": "n6_soft_coulomb_pilot",
        "claim": "The resource-capped N6,D10,K1-to-K4 soft-Coulomb pilot passes direct-CI, symmetry, variance, and resource gates.",
        "artifact": "docs/experiments/results/phase29_n6_soft_coulomb_pilot.json",
        "verifier_module": "scripts.verify_phase29_n6_soft_coulomb_pilot",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/benchmark_phase29_n6_soft_coulomb_pilot.py",
        "verify_command": "python scripts/verify_phase29_n6_soft_coulomb_pilot.py",
        "seeds": [29, 2914],
    },
    {
        "id": "n6_soft_coulomb_multiseed",
        "claim": "N6,D10,K4 soft-Coulomb FEMPS passes three fixed blind-seed stability gates.",
        "artifact": "docs/experiments/results/phase29_n6_multiseed_stability.json",
        "verifier_module": "scripts.verify_phase29_n6_multiseed_stability",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/benchmark_phase29_n6_multiseed_stability.py",
        "verify_command": "python scripts/verify_phase29_n6_multiseed_stability.py",
        "seeds": [31, 37, 43],
    },
    {
        "id": "matched_n4_n6_transition_cost",
        "claim": "At fixed D10,K4,L19, N4/N6 value-gradient kernels pass matched value, operation-count, and cost audits.",
        "artifact": "docs/experiments/results/phase29_n4_n6_matched_cost.json",
        "verifier_module": "scripts.verify_phase29_n4_n6_matched_cost",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/benchmark_phase29_n4_n6_matched_cost.py",
        "verify_command": "python scripts/verify_phase29_n4_n6_matched_cost.py",
        "seeds": [],
        "registered_tolerances": {
            "auto_minor_value_difference": 1e-10,
            "operator_factorization_error": 1e-11,
        },
    },
    {
        "id": "n6_independent_dk_convergence",
        "claim": "N6 soft-Coulomb FEMPS passes independent K1/K2/K4 and D8/D10/D12 convergence, operator, symmetry, and resource gates.",
        "artifact": "docs/experiments/results/phase32_n6_convergence.json",
        "verifier_module": "scripts.verify_phase32_n6_convergence",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/benchmark_phase32_n6_convergence.py",
        "verify_command": "python scripts/verify_phase32_n6_convergence.py",
        "seeds": [3201, 3202, 3204, 3284, 3212],
    },
    {
        "id": "n6_vectorized_transition_backend",
        "claim": "Batched N6,D10,K4 transitions pass reference and CPU/Blackwell value-gradient parity plus a matched 160/80 optimization gate.",
        "artifact": "docs/experiments/results/phase33_vectorized_transitions.json",
        "verifier_module": "scripts.verify_phase33_vectorized_transitions",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/benchmark_phase33_vectorized_transitions.py",
        "verify_command": "python scripts/verify_phase33_vectorized_transitions.py",
        "seeds": [3301, 3304],
    },
    {
        "id": "n6_adaptive_correlation_growth",
        "claim": "Truth-free N6,D12 K4-to-K6 adaptive growth passes nesting, CI, variance, symmetry, resource, and cold-K6 control gates.",
        "artifact": "docs/experiments/results/phase34_adaptive_k_growth.json",
        "verifier_module": "scripts.verify_phase34_adaptive_k_growth",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/benchmark_phase34_adaptive_k_growth.py",
        "verify_command": "python scripts/verify_phase34_adaptive_k_growth.py",
        "seeds": [3451, 3452, 3460],
    },
    {
        "id": "n6_adaptive_pool_stability",
        "claim": "Three fresh truth-free N6,D12 adaptive K5/K6 candidate-pool lineages pass stability gates; the automatic stop rule remains unadmitted.",
        "artifact": "docs/experiments/results/phase35_adaptive_pool_stability.json",
        "verifier_module": "scripts.verify_phase35_adaptive_pool_stability",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/benchmark_phase35_adaptive_pool_stability.py",
        "verify_command": "python scripts/verify_phase35_adaptive_pool_stability.py",
        "seeds": [3511, 3512, 3521, 3522, 3531, 3532],
    },
    {
        "id": "n6_public_adaptive_solver",
        "claim": "The public bounded adaptive API exactly reproduces frozen N6,D12 lineage-1 selections and energies across an explicit K5 interruption/resume boundary.",
        "artifact": "docs/experiments/results/phase36_public_adaptive_solver.json",
        "verifier_module": "scripts.verify_phase36_public_adaptive_solver",
        "verifier_argument": "path",
        "benchmark_command": "python scripts/benchmark_phase36_public_adaptive_solver.py",
        "verify_command": "python scripts/verify_phase36_public_adaptive_solver.py",
        "seeds": [3511, 3512],
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    entries = []
    for specification in ENTRIES:
        entry = dict(specification)
        artifact_path = Path(entry["artifact"])
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        entry.update(
            {
                "artifact_sha256": _sha256(artifact_path),
                "artifact_schema_version": payload["schema_version"],
                "evidence_level": payload["evidence_level"],
                "scientific_boundary": payload.get(
                    "scientific_boundary",
                    "bounded numerical evidence; see the linked report",
                ),
                "registered_tolerances": entry.get(
                    "registered_tolerances", payload.get("thresholds", {})
                ),
            }
        )
        entries.append(entry)
    manifest = {
        "schema_version": 1,
        "manifest": "phase30_femps_method_reproduction",
        "evidence_policy": "only independently verified committed numerical artifacts are admitted",
        "solver_contract": {
            "document": "docs/DIAGONAL_PATH_SOLVER_CONTRACT.md",
            "result_schema_version": DIAGONAL_PATH_RESULT_SCHEMA_VERSION,
            "checkpoint_schema_version": DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION,
            "adaptive_result_schema_version": (
                ADAPTIVE_DIAGONAL_PATH_RESULT_SCHEMA_VERSION
            ),
            "adaptive_checkpoint_schema_version": (
                ADAPTIVE_DIAGONAL_PATH_CHECKPOINT_SCHEMA_VERSION
            ),
        },
        "global_boundaries": [
            "restricted nonbranching diagonal-path FEMPS only",
            "first quantized continuous functional basis",
            "numerical artifacts are not theorems",
            "no generic contraction, asymptotic scaling, runtime superiority, or N8 claim",
            "direct CI remains faster in the current bounded truth region",
        ],
        "entries": entries,
    }
    output = Path("docs/experiments/results/phase30_reproduction_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "entries": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
