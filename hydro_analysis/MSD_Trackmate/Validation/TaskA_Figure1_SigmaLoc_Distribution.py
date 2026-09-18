"""
Figure 1 -- distribution of localization error sigma_loc from immobilized
particles.

Pure consumer: reads cache/localization_error_immobilized.pkl (produced by
TaskA_Immobilized_Compute.py). Never touches raw XML/TIFF.

Three figures:
  1) sigma_x/sigma_y (static estimator) box distributions, one panel per
     particle size (real DLS-measured size via core.io.get_dls_labels(),
     not the nominal folder name).
  2) For the 20 nm (35 nm-labeled) group specifically, the same box
     distributions split by dataset (the two immobilized folders were
     recorded on different days/setups -- this checks whether sigma_loc
     depends on which measurement it came from).
  3) Cross-check scatter: static-position estimator vs frame-to-frame-
     difference estimator (see TaskA_Immobilized_Compute.py's docstring for
     both definitions), with a 1:1 reference line -- both are reported, the
     static estimator is never silently replaced by whichever is smaller.
  Plus a scatter of sigma (static) vs track length, purely descriptive
  (no claim of a systematic trend unless one is visually obvious).

Run TaskA_Immobilized_Compute.py first (or after any raw-data change) to
refresh the cache.

Shows each figure first (plt.show(), blocking); only after the window is
closed does it save into Auswertungsbilder\\LocError_Immobilized_Distribution\\.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent.parent / "cache" / "localization_error_immobilized.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "LocError_Immobilized_Distribution"

COLOR_X, COLOR_X_DARK = "#0000da", "#000099"
COLOR_Y, COLOR_Y_DARK = "#da0000", "#990000"


def load_df() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst TaskA_Immobilized_Compute.py ausführen."
        )
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def _label(size_nm: float, dls_labels: dict[float, int]) -> str:
    return f"{dls_labels.get(size_nm, int(size_nm))} nm"


def _boxplot_by_group(ax, groups: list[np.ndarray], labels: list[str], positions: np.ndarray,
                       color: str, edge: str, width: float) -> None:
    bp = ax.boxplot(groups, positions=positions, widths=width, patch_artist=True,
                     showfliers=False, manage_ticks=False)
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_edgecolor(edge)
        box.set_alpha(0.6)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(edge)


def plot_by_size(df: pd.DataFrame, dls_labels: dict[float, int]) -> plt.Figure:
    sizes = sorted(df["particle_size_nm"].dropna().unique())
    labels = [_label(s, dls_labels) for s in sizes]
    x_groups = [df.loc[df["particle_size_nm"] == s, "sigma_x_static_nm"].dropna().to_numpy() for s in sizes]
    y_groups = [df.loc[df["particle_size_nm"] == s, "sigma_y_static_nm"].dropna().to_numpy() for s in sizes]

    positions = np.arange(len(sizes), dtype=float)
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    _boxplot_by_group(ax, x_groups, labels, positions - 0.18, COLOR_X, COLOR_X_DARK, 0.32)
    _boxplot_by_group(ax, y_groups, labels, positions + 0.18, COLOR_Y, COLOR_Y_DARK, 0.32)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\sigma_{loc}$ (nm, static estimator)")
    ax.set_xlabel("Particle size (nm)")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COLOR_X, edgecolor=COLOR_X_DARK, alpha=0.6, label=r"$\sigma_x$"),
        Patch(facecolor=COLOR_Y, edgecolor=COLOR_Y_DARK, alpha=0.6, label=r"$\sigma_y$"),
    ], loc="upper left", frameon=False)
    return fig


def plot_by_dataset(df: pd.DataFrame, dls_labels: dict[float, int], size_nm: float = 20.0) -> plt.Figure | None:
    sub = df[df["particle_size_nm"] == size_nm]
    if sub.empty:
        return None
    datasets = sorted(sub["dataset_label"].unique())
    x_groups = [sub.loc[sub["dataset_label"] == d, "sigma_x_static_nm"].dropna().to_numpy() for d in datasets]
    y_groups = [sub.loc[sub["dataset_label"] == d, "sigma_y_static_nm"].dropna().to_numpy() for d in datasets]

    positions = np.arange(len(datasets), dtype=float)
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    _boxplot_by_group(ax, x_groups, datasets, positions - 0.18, COLOR_X, COLOR_X_DARK, 0.32)
    _boxplot_by_group(ax, y_groups, datasets, positions + 0.18, COLOR_Y, COLOR_Y_DARK, 0.32)

    ax.set_xticks(positions)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel(r"$\sigma_{loc}$ (nm, static estimator)")
    ax.set_xlabel("Dataset")
    ax.set_title(f"{_label(size_nm, dls_labels)} particles, by measurement/dataset")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COLOR_X, edgecolor=COLOR_X_DARK, alpha=0.6, label=r"$\sigma_x$"),
        Patch(facecolor=COLOR_Y, edgecolor=COLOR_Y_DARK, alpha=0.6, label=r"$\sigma_y$"),
    ], loc="upper left", frameon=False)
    return fig


def plot_crosscheck(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    x_static = df["sigma_x_static_nm"].to_numpy()
    x_f2f = df["sigma_x_frame2frame_nm"].to_numpy()
    y_static = df["sigma_y_static_nm"].to_numpy()
    y_f2f = df["sigma_y_frame2frame_nm"].to_numpy()

    ax.scatter(x_static, x_f2f, s=14, alpha=0.35, facecolor=COLOR_X, edgecolor="none", label=r"$x$")
    ax.scatter(y_static, y_f2f, s=14, alpha=0.35, facecolor=COLOR_Y, edgecolor="none", label=r"$y$")

    finite = np.concatenate([x_static[np.isfinite(x_static)], y_static[np.isfinite(y_static)],
                              x_f2f[np.isfinite(x_f2f)], y_f2f[np.isfinite(y_f2f)]])
    lim = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
    ax.plot([0, lim], [0, lim], color="black", linewidth=1.0, linestyle="--", zorder=1, label="1:1")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"$\sigma_{loc}$, static estimator (nm)")
    ax.set_ylabel(r"$\sigma_{loc}$, frame-to-frame estimator (nm)")
    ax.legend(loc="upper left", frameon=False)
    return fig


def plot_vs_track_length(df: pd.DataFrame, dls_labels: dict[float, int]) -> plt.Figure:
    sizes = sorted(df["particle_size_nm"].dropna().unique())
    n = len(sizes)
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.15 / 3, nrows * 5.00 / 2),
                              constrained_layout=True, squeeze=False)
    ax_flat = axes.flatten()
    for ax in ax_flat[n:]:
        ax.set_visible(False)

    for ax, size_nm in zip(ax_flat, sizes):
        sub = df[df["particle_size_nm"] == size_nm]
        ax.scatter(sub["track_length_frames"], sub["sigma_x_static_nm"], s=10, alpha=0.35,
                   facecolor=COLOR_X, edgecolor="none", label=r"$\sigma_x$")
        ax.scatter(sub["track_length_frames"], sub["sigma_y_static_nm"], s=10, alpha=0.35,
                   facecolor=COLOR_Y, edgecolor="none", label=r"$\sigma_y$")
        ax.set_xscale("log")
        ax.set_xlabel("Track length (frames)")
        ax.set_ylabel(r"$\sigma_{loc}$ (nm)")
        ax.set_title(_label(size_nm, dls_labels))
        ax.minorticks_on()
    ax_flat[0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(r"$\sigma_{loc}$ vs. track length", fontsize=12, fontweight="semibold")
    return fig


def main() -> None:
    df = load_df()
    print(f"Cache geladen: {CACHE_FILE} ({len(df)} Tracks)")
    dls_labels = get_dls_labels()

    summary = df.groupby(["particle_size_nm", "dataset_label"]).agg(
        n_tracks=("particle_id", "count"),
        sigma_x_mean_nm=("sigma_x_static_nm", "mean"),
        sigma_x_std_nm=("sigma_x_static_nm", "std"),
        sigma_y_mean_nm=("sigma_y_static_nm", "mean"),
        sigma_y_std_nm=("sigma_y_static_nm", "std"),
        sigma_x_f2f_mean_nm=("sigma_x_frame2frame_nm", "mean"),
        sigma_y_f2f_mean_nm=("sigma_y_frame2frame_nm", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))

    with plt.rc_context(_RC):
        figs = {
            "sigma_loc_by_size.png": plot_by_size(df, dls_labels),
            "sigma_loc_by_dataset_20nm.png": plot_by_dataset(df, dls_labels, size_nm=20.0),
            "sigma_loc_crosscheck_static_vs_f2f.png": plot_crosscheck(df),
            "sigma_loc_vs_track_length.png": plot_vs_track_length(df, dls_labels),
        }
        for filename, fig in figs.items():
            if fig is None:
                continue
            plt.figure(fig.number)
            plt.show()
            if SAVE_PATH is not None:
                SAVE_PATH.mkdir(parents=True, exist_ok=True)
                png_path = SAVE_PATH / filename
                fig.savefig(png_path, dpi=600, bbox_inches="tight")
                print(f"Plot gespeichert: {png_path}")

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = SAVE_PATH / "sigma_loc_summary_by_size_dataset.csv"
        summary.to_csv(csv_path, index=False)
        print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
