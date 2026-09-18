"""
Histogram of per-particle track length (in frames), gridded by particle
size -- separate figures for D0 (water) and D_eff (20 mg/mL hydrogel).

Pure consumer: reads msd_d0_results.pkl (MSD_FromTrackmate_D0.py) and
msd_20mg_files.pkl (MSD_FromTrackmate_20mg.py). For each particle size,
track length is tracks_df.groupby('particle').size() (frames per
trajectory) pooled across every file of that size -- the same idiom already
used elsewhere in this repo (e.g. MSD_Diffusion_Correlations_Test_D0.py).
No re-filtering, no re-fitting: only counts rows already present in the
cached tracks_df.

Two separate figures (D0 and D_eff kept apart rather than overlaid, since
their track-length distributions differ enough that overlaying obscured
D_eff), each laid out as one panel per particle size (dynamic grid per
../Auswertung_von_iMSD.py's _plot_size_histograms: <=3 sizes -> 1 row, >3 ->
ceil(n/3) rows of 3 columns). Every panel shares a fixed x-axis (0-200
frames) so panels are directly comparable at a glance. Panel titles use the
real DLS particle size (core.io.get_dls_labels()), not the nominal folder
name, and show no N -- just the size.

Run MSD_FromTrackmate_D0.py and MSD_FromTrackmate_20mg.py first (or after
any raw-data change) to refresh both pickles.

Shows each figure first (plt.show(), blocking); only after the window is
closed does it save into the shared Auswertungsbilder\\TrackLength_Histogram\\
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
CACHE_D0   = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
CACHE_DEFF = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "TrackLength_Histogram"

X_MIN, X_MAX = 0.0, 200.0
N_BINS = 30
STYLE_D0   = dict(face="#009900", edge="#006600", label="D0 (water)", dataset="d0")
STYLE_DEFF = dict(face="#0000da", edge="#000099", label="D_eff (20 mg/mL)", dataset="deff")


def _load_cache(path: Path, compute_script: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _pool_track_lengths_by_size(results: dict) -> dict[float, np.ndarray]:
    """{particle_size_nm: array of per-trajectory lengths in frames}, pooled across all files of that size."""
    size_groups: dict[float, list] = {}
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        tracks_df = r.get("tracks_df")
        if size_nm is None or tracks_df is None or tracks_df.empty:
            continue
        lengths = tracks_df.groupby("particle").size().to_numpy()
        size_groups.setdefault(float(size_nm), []).append(lengths)
    return {s: np.concatenate(v) for s, v in size_groups.items()}


def _plot_grid(by_size: dict[float, np.ndarray], style: dict, dls_labels: dict[float, int]) -> plt.Figure:
    sizes = sorted(by_size.keys())
    n = len(sizes)
    if n == 0:
        raise ValueError(f"No track-length data found for {style['label']}.")

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

    for ax, size_nm in zip(ax_flat, sizes):
        vals = by_size[size_nm]
        label_nm = dls_labels.get(size_nm, int(size_nm))

        ax.hist(vals, bins=N_BINS, range=(X_MIN, X_MAX), color=style["face"], edgecolor=style["edge"],
                linewidth=0.4, alpha=0.85)

        ax.set_xlim(X_MIN, X_MAX)
        ax.set_xlabel("Track length (frames)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label_nm} nm")
        ax.minorticks_on()

    fig.suptitle(f"Track length per particle, by particle size -- {style['label']}",
                 fontsize=12, fontweight="semibold")
    return fig


def main() -> None:
    d0_results = _load_cache(CACHE_D0, "MSD_FromTrackmate_D0.py")
    print(f"Cache geladen: {CACHE_D0} ({len(d0_results)} Dateien)")
    deff_results = _load_cache(CACHE_DEFF, "MSD_FromTrackmate_20mg.py")
    print(f"Cache geladen: {CACHE_DEFF} ({len(deff_results)} Dateien)")

    dls_labels = get_dls_labels()
    d0_by_size = _pool_track_lengths_by_size(d0_results)
    deff_by_size = _pool_track_lengths_by_size(deff_results)

    with plt.rc_context(_RC):
        for by_size, style, filename in (
            (d0_by_size, STYLE_D0, "track_length_histogram_d0.png"),
            (deff_by_size, STYLE_DEFF, "track_length_histogram_deff.png"),
        ):
            fig = _plot_grid(by_size, style, dls_labels)
            plt.show()

            if SAVE_PATH is not None:
                SAVE_PATH.mkdir(parents=True, exist_ok=True)
                png_path = SAVE_PATH / filename
                fig.savefig(png_path, dpi=600, bbox_inches="tight")
                print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
