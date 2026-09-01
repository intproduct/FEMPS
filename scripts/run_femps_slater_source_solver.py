"""Run FEMPS from a canonical Slater source with an explicit finite K cap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from femps.algorithms import (
    load_slater_source_command_config,
    run_slater_source_adaptive_solver,
    validate_slater_source_result,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a canonical continuous-basis Slater determinant and run "
            "bounded adaptive diagonal-path FEMPS growth."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-k", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-source-steps-this-call", type=int)
    parser.add_argument("--max-adaptive-stages-this-call", type=int)
    args = parser.parse_args()

    config, record = load_slater_source_command_config(args.config)
    if args.max_k != config.max_terms:
        raise ValueError("--max-k must equal the finite maximum frozen in --config")
    if args.max_k <= 1:
        raise ValueError("--max-k must be a finite integer greater than one")
    if not args.resume and args.output.exists():
        raise ValueError("output already exists; use a new path or --resume")

    result = run_slater_source_adaptive_solver(
        config,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        max_source_steps_this_call=args.max_source_steps_this_call,
        max_adaptive_stages_this_call=args.max_adaptive_stages_this_call,
    )
    result["command"] = {
        "config_path": str(args.config),
        "config_sha256": _sha256(args.config),
        "checkpoint_path": str(args.checkpoint),
        "output_path": str(args.output),
        "requested_max_k": args.max_k,
        "registered_checkpoint_path": record["checkpoint_path"],
        "registered_output_path": record["output_path"],
    }
    validate_slater_source_result(result)
    _write_atomic(args.output, result)
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "resumed": result["resumed"],
                "current_terms": result["current_terms"],
                "energies": [
                    stage["optimizer_result"]["energy"] for stage in result["stages"]
                ],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
