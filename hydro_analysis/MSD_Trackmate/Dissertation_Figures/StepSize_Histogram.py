"""
Histogram of per-step Euclidean displacement (nm), gridded by particle
size, for D0 (water) and D_eff (20 mg/mL hydrogel) overlaid per panel.

Pure consumer: reads stepsize_d0_results.pkl (Schrittweiten_methode_D0.py)
and stepsize_20mg_results.pkl (Schrittweiten_methode_20mg.py). For each
particle size, pools sqrt(dx_nm**2 + dy_nm**2) across every step in every
file of that size, where dx_nm/dy_nm = fit_results_step["step_df"]["dx"/"dy"]
(pixels) * mpp (µm/px, per-file) * 1000 -- same pixel->nm conversion already
used in core.visualization.plot_step_size_overlay() and
plot_dx_dy_distributions(). No re-fitting.

One combined figure, one panel per particle size (same 2x3/dynamic grid
layout as TrackLength_Histogram.py), D0 and D_eff overlaid per panel (two
colors, shared bin edges), mirroring the "for D_0 and D_eff" request as one
direct-comparison view rather than two separate grids.

Run Schrittweiten_methode_D0.py and Schrittweiten_methode_20mg.py first (or
after any raw-data change) to refresh both pickles.

Shows the figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\StepSize_Histogram\\
subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_D0   = Path(__file__).parent.parent / "cache" / "stepsize_d0_results.pkl"
CACHE_DEFF = Path(__file__).parent.parent / "cache" / "stepsize_20mg_results.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "StepSize_Histogram"

X_MIN, X_MAX = 0.0, 4000.0
N_BINS = 45   # Style_guide.txt §6: 25-60 bins, "eher mehr aber nicht kammartig"
STYLE_D0   = dict(face="#009900", edge="#006600", label="D0 (water)")
STYLE_DEFF = dict(face="#0000da", edge="#000099", label="D_eff (20 mg/mL)")


def _load_cache(path: Path, compute_script: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _pool_step_sizes_by_size(results: dict) -> dict[float, np.ndarray]:
    """{particle_size_nm: array of Euclidean step sizes (nm)}, pooled across all files of that size."""
    size_groups: dict[float, list] = {}
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        mpp = r.get("mpp")
        step_result = r.get("fit_results_step")
        step_df = step_result.get("step_df") if step_result else None
        if size_nm is None or mpp is None or step_df is None or step_df.empty:
            continue
        steps_nm = np.hypot(step_df["dx"].to_numpy() * mpp * 1000.0,
                             step_df["dy"].to_numpy() * mpp * 1000.0)
        size_groups.setdefault(float(size_nm), []).append(steps_nm)
    return {s: np.concatenate(v) for s, v in size_groups.items()}


def _plot_grid(d0_by_size: dict[float, np.ndarray], deff_by_size: dict[float, np.ndarray],
               dls_labels: dict[float, int]) -> plt.Figure:
    sizes = sorted(set(d0_by_size) | set(deff_by_size))
    n = len(sizes)
    if n == 0:
        raise ValueError("No step-size data found in either cache.")

    if n <= 3:
        nrows, ncols = 1, n
    else:
        ncols = 3
        nrows = int(np.ceil(n / ncols))

    subplot_w = 7.15 / 3
    subplot_h = 5.00 / 2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * subplot_w, nrows * subplot_h),
        constrained_layout=True,
        squeeze=False,
    )
    ax_flat = axes.flatten()
    for ax in ax_flat[n:]:
        ax.set_visible(False)

    bins = np.linspace(X_MIN, X_MAX, N_BINS + 1)

    for ax, size_nm in zip(ax_flat, sizes):
        d0_vals   = d0_by_size.get(size_nm, np.array([]))
        deff_vals = deff_by_size.get(size_nm, np.array([]))
        label_nm = dls_labels.get(size_nm, int(size_nm))
        if d0_vals.size == 0 and deff_vals.size == 0:
            ax.text(0.5, 0.5, "keine Daten", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{label_nm} nm")
            continue

        if d0_vals.size:
            ax.hist(d0_vals, bins=bins, alpha=0.5, color=STYLE_D0["face"], edgecolor=STYLE_D0["edge"],
                    linewidth=0.4, label=STYLE_D0["label"])
        if deff_vals.size:
            ax.hist(deff_vals, bins=bins, alpha=0.5, color=STYLE_DEFF["face"], edgecolor=STYLE_DEFF["edge"],
                    linewidth=0.4, label=STYLE_DEFF["label"])

        ax.set_xlim(X_MIN, X_MAX)
        ax.set_xlabel("Step size (nm)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label_nm} nm")
        ax.minorticks_on()

    # Legend placed inside the first panel (not figure-level) to avoid colliding
    # with the suptitle -- colors are the same in every panel.
    ax_flat[0].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=STYLE_D0["face"], edgecolor=STYLE_D0["edge"], alpha=0.5, label=STYLE_D0["label"]),
            plt.Rectangle((0, 0), 1, 1, facecolor=STYLE_DEFF["face"], edgecolor=STYLE_DEFF["edge"], alpha=0.5, label=STYLE_DEFF["label"]),
        ],
        loc="upper right", fontsize=7, frameon=False,
    )
    fig.suptitle("Step size per particle size, D0 vs. D_eff", fontsize=12, fontweight="semibold")
    return fig


def main() -> None:
    d0_results = _load_cache(CACHE_D0, "Schrittweiten_methode_D0.py")
    print(f"Cache geladen: {CACHE_D0} ({len(d0_results)} Dateien)")
    deff_results = _load_cache(CACHE_DEFF, "Schrittweiten_methode_20mg.py")
    print(f"Cache geladen: {CACHE_DEFF} ({len(deff_results)} Dateien)")

    dls_labels = get_dls_labels()
    d0_by_size = _pool_step_sizes_by_size(d0_results)
    deff_by_size = _pool_step_sizes_by_size(deff_results)

    with plt.rc_context(_RC):
        fig = _plot_grid(d0_by_size, deff_by_size, dls_labels)
        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / "step_size_histogram_d0_deff.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
