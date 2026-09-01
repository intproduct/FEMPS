"""Build paper-ready FEMPS method figures from manifest-hashed artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MANIFEST = Path("docs/experiments/results/phase30_reproduction_manifest.json")
FIGURE_DIR = Path("docs/paper/figures")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sources() -> tuple[dict, dict[str, dict]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = {}
    for entry in manifest["entries"]:
        path = Path(entry["artifact"])
        if _sha256(path) != entry["artifact_sha256"]:
            raise ValueError(f"manifest source hash mismatch: {entry['id']}")
        sources[entry["id"]] = json.loads(path.read_text(encoding="utf-8"))
    return manifest, sources


def _save_figure(figure: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "FEMPS Phase 30 artifact builder"},
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "FEMPS Phase 30 artifact builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return [png, pdf]


def _convergence_figure(sources: dict[str, dict]) -> tuple[list[Path], dict]:
    transfer = sources["n4_soft_coulomb_transferability"]
    basis = sources["n4_soft_coulomb_basis_extension"]
    high_k = sources["n4_soft_coulomb_high_basis_correlation"]
    n6_pilot = sources["n6_soft_coulomb_pilot"]
    n6_stability = sources["n6_soft_coulomb_multiseed"]

    dense_d6 = transfer["dense_ci_comparators"][0]["energy"]
    k_axis = transfer["convergence"]["K_axis_D6"]
    k_values = [point["K"] for point in k_axis]
    k_errors = [point["energy"] - dense_d6 for point in k_axis]

    d6_d8 = transfer["convergence"]["D_axis_K4"]
    d8_d12 = basis["convergence"]["D_axis_K4"]
    d_axis = [d6_d8[0], *d8_d12]
    d_values = [point["D"] for point in d_axis]
    d_errors = [point["absolute_error_vs_D14"] for point in d_axis]

    n6_k1 = n6_pilot["points"][0]["error_vs_dense_quadrature_ci"]
    n6_continuation = n6_pilot["points"][1]["error_vs_dense_quadrature_ci"]
    n6_blind = [
        point["error_vs_dense_quadrature_ci"] for point in n6_stability["points"]
    ]

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), constrained_layout=True)
    blue, orange, green = "#3572B0", "#E6862A", "#2A9D6F"

    axes[0].plot(k_values, k_errors, marker="o", color=blue, linewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xticks(k_values)
    axes[0].set_xlabel("Correlation terms K")
    axes[0].set_ylabel("Error vs same-basis CI")
    axes[0].set_title("(a) N=4, D=6 correlation")

    axes[1].plot(d_values, d_errors, marker="s", color=orange, linewidth=1.5)
    axes[1].set_yscale("log")
    axes[1].set_xticks(d_values)
    axes[1].set_xlabel("Functional-basis order D")
    axes[1].set_ylabel("|E - E(D=14 reference)|")
    axes[1].set_title("(b) N=4, K=4 basis")

    labels = ["K1\nblind", "K4\ncontinuation", "K4\nseed 31", "K4\nseed 37", "K4\nseed 43"]
    values = [n6_k1, n6_continuation, *n6_blind]
    colors = [orange, green, blue, blue, blue]
    axes[2].bar(np.arange(len(values)), values, color=colors, width=0.72)
    axes[2].set_yscale("log")
    axes[2].set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
    axes[2].set_ylabel("Error vs same-basis CI")
    axes[2].set_title("(c) N=6, D=10 correlation")

    for axis in axes:
        axis.grid(axis="y", linewidth=0.5, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Restricted FEMPS convergence (numerical evidence)", fontsize=11)
    paths = _save_figure(figure, "femps-convergence-summary")
    data = {
        "N4_D6_K_axis": {"K": k_values, "error_vs_same_basis_CI": k_errors},
        "N4_K4_D_axis": {"D": d_values, "absolute_error_vs_D14": d_errors},
        "N6_D10_K_comparison": {"labels": labels, "error_vs_same_basis_CI": values},
        "D14_role": "finite-basis numerical reference, not a continuum theorem",
    }
    return paths, data


def _cost_figure(sources: dict[str, dict]) -> tuple[list[Path], dict]:
    cost = sources["matched_n4_n6_transition_cost"]
    n4, n6 = cost["points"]
    modes = [
        "auto_forward",
        "auto_forward_backward",
        "minor_forward",
        "minor_forward_backward",
    ]
    mode_labels = ["auto\nvalue", "auto\nvalue+grad", "minor\nvalue", "minor\nvalue+grad"]
    n4_times = [n4["modes"][mode]["median_seconds"] for mode in modes]
    n6_times = [n6["modes"][mode]["median_seconds"] for mode in modes]

    center_ranks = {
        "N4": [
            n4["femps_ordinary_particle_tt_ranks"][1],
            n4["dense_ci_ordinary_particle_tt_ranks"][1],
        ],
        "N6": [
            n6["femps_ordinary_particle_tt_ranks"][2],
            n6["dense_ci_ordinary_particle_tt_ranks"][2],
        ],
    }

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), constrained_layout=True)
    blue, orange = "#3572B0", "#E6862A"
    x = np.arange(len(modes))
    width = 0.36
    axes[0].bar(x - width / 2, n4_times, width, label="N=4", color=blue)
    axes[0].bar(x + width / 2, n6_times, width, label="N=6", color=orange)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, mode_labels)
    axes[0].set_ylabel("Median CPU time (s)")
    axes[0].set_title("(a) Fixed D=10, K=4, L=19")
    axes[0].legend(frameon=False)

    rank_x = np.arange(2)
    axes[1].bar(
        rank_x - width / 2,
        [center_ranks["N4"][0], center_ranks["N6"][0]],
        width,
        label="FEMPS (K=4)",
        color=blue,
    )
    axes[1].bar(
        rank_x + width / 2,
        [center_ranks["N4"][1], center_ranks["N6"][1]],
        width,
        label="Dense CI",
        color=orange,
    )
    axes[1].set_xticks(rank_x, ["N=4", "N=6"])
    axes[1].set_ylabel("Ordinary particle-TT center rank")
    axes[1].set_title("(b) Exchange carrier at fixed K")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", linewidth=0.5, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Restricted FEMPS structure and cost (numerical evidence)", fontsize=11)
    paths = _save_figure(figure, "femps-structure-cost-summary")
    data = {
        "timing_modes": modes,
        "N4_median_seconds": n4_times,
        "N6_median_seconds": n6_times,
        "ordinary_particle_TT_center_ranks": center_ranks,
        "fixed_correlation_terms_K": 4,
    }
    return paths, data


def main() -> None:
    manifest, sources = _load_sources()
    convergence_paths, convergence_data = _convergence_figure(sources)
    cost_paths, cost_data = _cost_figure(sources)
    all_paths = [*convergence_paths, *cost_paths]
    source_hashes = {
        entry["id"]: entry["artifact_sha256"] for entry in manifest["entries"]
    }
    provenance = {
        "schema_version": 1,
        "evidence_level": "numerical",
        "scientific_boundary": "paper figures from twelve manifest-hashed artifacts; no asymptotic or superiority claim",
        "manifest": str(MANIFEST),
        "manifest_sha256": _sha256(MANIFEST),
        "source_artifact_sha256": source_hashes,
        "figures": {str(path): _sha256(path) for path in all_paths},
        "plotted_data": {
            "femps-convergence-summary": convergence_data,
            "femps-structure-cost-summary": cost_data,
        },
    }
    output = FIGURE_DIR / "phase30-method-figure-provenance.json"
    output.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": len(all_paths), "provenance": str(output)}, indent=2))


if __name__ == "__main__":
    main()
