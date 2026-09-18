"""
Figure 4 -- D and n vs. individual trajectory length.

Pure consumer: reads cache/track_length_bias_d0.pkl and
cache/track_length_bias_20mg.pkl (TaskB1_TrackLengthBias_Compute.py). Never
touches raw XML/TIFF or the msd_*.pkl caches directly.

For each condition (water/hydrogel) and particle size: D vs. track length
and n vs. track length scatter (log-x, since track length spans a wide
range), with median + interquartile-range per length bin and the number of
tracks per bin annotated. Also reports Pearson and Spearman correlation of
D (and n) vs log10(track length) -- descriptive statistics, not a claim of
a systematic bias unless the trend is visually and statistically clear.
This realizes B4 (condition/size split) without a separate script, since
every row already carries both fields.

Run TaskB1_TrackLengthBias_Compute.py first (or after any raw-data change)
to refresh both inputs.

Shows each figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\TrackLengthBias_Scatter\\
subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_D0 = Path(__file__).parent.parent / "cache" / "track_length_bias_d0.pkl"
CACHE_20MG = Path(__file__).parent.parent / "cache" / "track_length_bias_20mg.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "TrackLengthBias_Scatter"

BIN_EDGES = np.array([5, 10, 15, 20, 30, 50, 100, 200, 500, 1000])
TARGET_SIZES_NM = (20.0, 50.0)

COLOR_WATER, COLOR_WATER_DARK = "#3B6E8C", "#2A4F66"
COLOR_GEL, COLOR_GEL_DARK = "#da0000", "#990000"


def _load_pickle(path: Path, compute_script: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _binned_stats(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (x >= lo) & (x < hi)
        if not mask.any():
            continue
        vals = y[mask]
        rows.append({
            "bin_lo": lo, "bin_hi": hi, "n": int(mask.sum()),
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
        })
    return pd.DataFrame(rows)


def plot_vs_length(df: pd.DataFrame, value_col: str, ylabel: str, color: str, color_dark: str,
                    condition: str, size_nm: float, label_nm: int) -> tuple[plt.Figure, dict]:
    x = df["track_length_frames"].to_numpy(dtype=float)
    y = df[value_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    ax.scatter(x, y, s=10, alpha=0.3, facecolor=color, edgecolor="none", label="individual tracks")

    binned = _binned_stats(x, y)
    if not binned.empty:
        centers = np.sqrt(binned["bin_lo"] * binned["bin_hi"])
        yerr = np.vstack([binned["median"] - binned["q25"], binned["q75"] - binned["median"]])
        ax.errorbar(centers, binned["median"], yerr=yerr, fmt="o-", color=color_dark,
                    markersize=6, linewidth=1.6, capsize=3, label="median ± IQR per bin")
        for cx, cy, n in zip(centers, binned["median"], binned["n"]):
            ax.annotate(f"n={n}", (cx, cy), textcoords="offset points", xytext=(0, 8),
                       fontsize=7, ha="center", color=color_dark)

    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    stats_out = {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    if valid.sum() >= 3:
        log_x = np.log10(x[valid])
        pear = scipy_stats.pearsonr(log_x, y[valid])
        spear = scipy_stats.spearmanr(log_x, y[valid])
        stats_out = {"pearson_r": float(pear[0]), "pearson_p": float(pear[1]),
                     "spearman_r": float(spear.correlation), "spearman_p": float(spear.pvalue)}

    ax.set_xscale("log")
    ax.set_xlabel("Track length (frames)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{label_nm} nm, {condition}  |  N={len(df)} tracks\n"
                 f"Pearson r={stats_out['pearson_r']:.3f} (p={stats_out['pearson_p']:.2g}), "
                 f"Spearman r={stats_out['spearman_r']:.3f} (p={stats_out['spearman_p']:.2g})  "
                 f"vs. log10(length)", fontsize=9)
    ax.legend(loc="best", fontsize=8, frameon=False)
    return fig, stats_out


def main() -> None:
    df_water = _load_pickle(CACHE_D0, "TaskB1_TrackLengthBias_Compute.py")
    df_gel = _load_pickle(CACHE_20MG, "TaskB1_TrackLengthBias_Compute.py")
    print(f"Water: {len(df_water)} Tracks, Hydrogel: {len(df_gel)} Tracks")

    dls_labels = get_dls_labels()
    corr_rows = []

    with plt.rc_context(_RC):
        for df, condition, color, color_dark in (
            (df_water, "water", COLOR_WATER, COLOR_WATER_DARK),
            (df_gel, "hydrogel", COLOR_GEL, COLOR_GEL_DARK),
        ):
            sizes = sorted(set(df["particle_size_nm"].dropna().unique()) & set(TARGET_SIZES_NM)) \
                if condition == "hydrogel" else sorted(df["particle_size_nm"].dropna().unique())
            for size_nm in sizes:
                sub = df[df["particle_size_nm"] == size_nm]
                if sub.empty:
                    continue
                label_nm = dls_labels.get(size_nm, int(size_nm))

                for value_col, ylabel, tag in (
                    ("D_um2_per_s", r"$D$ (µm²/s)", "D"),
                    ("exponent", r"$n$", "n"),
                ):
                    fig, stats_out = plot_vs_length(sub, value_col, ylabel, color, color_dark,
                                                     condition, size_nm, label_nm)
                    plt.show()
                    if SAVE_PATH is not None:
                        SAVE_PATH.mkdir(parents=True, exist_ok=True)
                        fname = f"{tag}_vs_length_{condition}_{int(size_nm)}nm.png"
                        fig.savefig(SAVE_PATH / fname, dpi=600, bbox_inches="tight")
                        print(f"Plot gespeichert: {SAVE_PATH / fname}")
                    corr_rows.append({"condition": condition, "particle_size_nm": size_nm,
                                       "label_nm": label_nm, "value": tag, "n_tracks": len(sub), **stats_out})

    corr_table = pd.DataFrame(corr_rows)
    print(corr_table.to_string(index=False))
    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = SAVE_PATH / "track_length_correlation_stats.csv"
        corr_table.to_csv(csv_path, index=False)
        print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
