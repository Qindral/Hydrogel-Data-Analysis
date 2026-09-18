"""
Histogram of individual-particle (per-track) diffusion coefficient D,
gridded by particle size -- separate figures for D0 (water) and D_eff
(20 mg/mL hydrogel).

Pure consumer: reads msd_d0_results.pkl (MSD_FromTrackmate_D0.py) and
msd_20mg_files.pkl (MSD_FromTrackmate_20mg.py).

"Individual" here means per TRACK, not per file: for every particle size,
each individual trajectory's own iMSD curve (fit_results_MSD["imsd"], one
column per track) is linear-fit (MSD = 4D*tau + c over the first
DEFAULT_MSD_FIT_POINTS lag points) to get that one particle's own D --
mirrors ../Auswertung_von_iMSD.py's _fit_track_D / _extract_track_D_per_size
exactly (duplicated here rather than imported, since Auswertung_von_iMSD.py
mutates mpl.rcParams at import time as a module-level side effect, which
would leak into this script's own plt.rc_context(_RC) styling). No
re-fitting of anything else, no data pooled across files beyond what the
cache already contains per track.

Two separate figures (D0 and D_eff), each one panel per particle size
(dynamic grid per Auswertung_von_iMSD.py's layout), log-x axis with
np.logspace bin edges (same precedent). Panel titles use the real DLS
particle size (core.io.get_dls_labels()), not the nominal folder name, and
show no N. No reference lines are drawn inside the histograms.

Run MSD_FromTrackmate_D0.py and MSD_FromTrackmate_20mg.py first (or after
any raw-data change) to refresh both pickles.

Shows each figure first (plt.show(), blocking); only after the window is
closed does it save into the shared Auswertungsbilder\\Deff_Histogram\\
subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.core.analysis import DEFAULT_MSD_FIT_POINTS
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_D0   = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
CACHE_DEFF = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "Deff_Histogram"

N_BINS = 30
STYLE_D0   = dict(face="#009900", edge="#006600", label="D0 (water), per track")
STYLE_DEFF = dict(face="#0000da", edge="#000099", label="D_eff (20 mg/mL), per track")


def _load_cache(path: Path, compute_script: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _fit_track_D(lag_times: np.ndarray, msd_values: np.ndarray, n_points: int) -> float:
    """Linear fit MSD = 4D*tau + c for one individual track. NaN if the fit fails or D <= 0."""
    mask = (
        np.isfinite(lag_times) & np.isfinite(msd_values)
        & (lag_times > 0) & (msd_values > 0)
    )
    lt = lag_times[mask][:n_points]
    mv = msd_values[mask][:n_points]
    if len(lt) < 2:
        return np.nan
    slope, _ = np.polyfit(lt, mv, 1)
    D = slope / 4.0
    return float(D) if D > 0 else np.nan


def _extract_track_D_per_size(results: dict, n_points: int) -> dict[float, np.ndarray]:
    """{particle_size_nm: array of per-track D values (µm²/s)}, one entry per individual trajectory."""
    size_D: dict[float, list] = {}
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        fit = r.get("fit_results_MSD")
        if size_nm is None or fit is None:
            continue
        imsd: pd.DataFrame | None = fit.get("imsd")
        if imsd is None or imsd.empty:
            continue
        lag_times = np.asarray(imsd.index.values, dtype=float)
        for col in imsd.columns:
            msd_vals = np.asarray(imsd[col].values, dtype=float)
            D = _fit_track_D(lag_times, msd_vals, n_points)
            if np.isfinite(D):
                size_D.setdefault(float(size_nm), []).append(D)
    return {s: np.array(v) for s, v in size_D.items()}


def _plot_grid(size_D: dict[float, np.ndarray], style: dict, dls_labels: dict[float, int]) -> plt.Figure:
    sizes = sorted(size_D.keys())
    n = len(sizes)
    if n == 0:
        raise ValueError(f"No per-track D data found for {style['label']}.")

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
        D_vals = size_D[size_nm]
        D_pos = D_vals[D_vals > 0]
        label_nm = dls_labels.get(size_nm, int(size_nm))

        if D_pos.size < 2:
            ax.text(0.5, 0.5, "keine Daten", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{label_nm} nm")
            continue

        bins = np.logspace(np.log10(D_pos.min()), np.log10(D_pos.max()), N_BINS + 1)
        ax.hist(D_pos, bins=bins, color=style["face"], edgecolor=style["edge"],
                linewidth=0.5, alpha=0.85)

        ax.set_xscale("log")
        ax.set_xlabel(r"$D$ (µm²/s)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label_nm} nm")
        ax.minorticks_on()

    fig.suptitle(f"Individual-particle D, by particle size -- {style['label']}",
                 fontsize=12, fontweight="semibold")
    return fig


def main() -> None:
    d0_results = _load_cache(CACHE_D0, "MSD_FromTrackmate_D0.py")
    print(f"Cache geladen: {CACHE_D0} ({len(d0_results)} Dateien)")
    deff_results = _load_cache(CACHE_DEFF, "MSD_FromTrackmate_20mg.py")
    print(f"Cache geladen: {CACHE_DEFF} ({len(deff_results)} Dateien)")
    dls_labels = get_dls_labels()

    d0_size_D = _extract_track_D_per_size(d0_results, DEFAULT_MSD_FIT_POINTS)
    deff_size_D = _extract_track_D_per_size(deff_results, DEFAULT_MSD_FIT_POINTS)

    with plt.rc_context(_RC):
        for size_D, style, filename in (
            (d0_size_D, STYLE_D0, "d0_individual_d_histogram.png"),
            (deff_size_D, STYLE_DEFF, "deff_individual_d_histogram.png"),
        ):
            fig = _plot_grid(size_D, style, dls_labels)
            plt.show()

            if SAVE_PATH is not None:
                SAVE_PATH.mkdir(parents=True, exist_ok=True)
                png_path = SAVE_PATH / filename
                fig.savefig(png_path, dpi=600, bbox_inches="tight")
                print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
