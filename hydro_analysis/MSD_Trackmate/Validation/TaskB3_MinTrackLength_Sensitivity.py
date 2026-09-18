"""
Sensitivity of ensemble D/n to the minimum-track-length threshold.

Pure consumer: reads cache/track_length_bias_d0.pkl and
cache/track_length_bias_20mg.pkl (TaskB1_TrackLengthBias_Compute.py). Never
touches raw XML/TIFF.

For a sweep of minimum-track-length thresholds, filters B1's per-track rows
to track_length_frames >= threshold and averages the ALREADY-COMPUTED
per-track D/n above that threshold. This is a documented, cheaper
APPROXIMATION of "raise core.analysis.MIN_TRACK_LENGTH / rerun
tp.filter_stubs + tp.imsd/tp.emsd + refit the ensemble eMSD from scratch at
each threshold": it reuses B1's independent per-track power-law fits rather
than reproducing trackpy's own count-weighted ensemble-eMSD pooling per
threshold, and is therefore NOT a substitute for the project's actual
headline ensemble D (from MSD_FromTrackmate_D0.py / _20mg.py). It is a
legitimate diagnostic sweep specifically because thresholds here are an
explicit, documented, symmetric comparison across the whole dataset -- not
a silent, one-off removal of inconvenient trajectories.

Run TaskB1_TrackLengthBias_Compute.py first (or after any raw-data change)
to refresh both inputs.

Shows the figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\MinTrackLength_Sensitivity\\
subfolder.
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
CACHE_D0 = Path(__file__).parent.parent / "cache" / "track_length_bias_d0.pkl"
CACHE_20MG = Path(__file__).parent.parent / "cache" / "track_length_bias_20mg.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "MinTrackLength_Sensitivity"

THRESHOLDS = [5, 10, 15, 20, 30, 50]
TARGET_SIZES_NM = (20.0, 50.0)

COLOR_WATER, COLOR_WATER_DARK = "#3B6E8C", "#2A4F66"
COLOR_GEL, COLOR_GEL_DARK = "#da0000", "#990000"


def _load_pickle(path: Path, compute_script: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def sensitivity_table(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    rows = []
    for size_nm in sorted(set(df["particle_size_nm"].dropna().unique()) & set(TARGET_SIZES_NM)):
        sub_size = df[df["particle_size_nm"] == size_nm]
        for threshold in THRESHOLDS:
            sub = sub_size[sub_size["track_length_frames"] >= threshold]
            if sub.empty:
                rows.append({"condition": condition, "particle_size_nm": size_nm, "threshold": threshold,
                             "n_tracks": 0, "D_mean": np.nan, "D_median": np.nan,
                             "n_mean": np.nan, "n_median": np.nan})
                continue
            rows.append({
                "condition": condition, "particle_size_nm": size_nm, "threshold": threshold,
                "n_tracks": len(sub),
                "D_mean": float(sub["D_um2_per_s"].mean()), "D_median": float(sub["D_um2_per_s"].median()),
                "n_mean": float(sub["exponent"].mean()), "n_median": float(sub["exponent"].median()),
            })
    return pd.DataFrame(rows)


def plot_sensitivity(table: pd.DataFrame, dls_labels: dict[float, int]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.00), constrained_layout=True)
    style = {"water": (COLOR_WATER, COLOR_WATER_DARK, "o"), "hydrogel": (COLOR_GEL, COLOR_GEL_DARK, "s")}

    for (condition, size_nm), g in table.groupby(["condition", "particle_size_nm"]):
        g = g.sort_values("threshold")
        color, color_dark, marker = style[condition]
        label_nm = dls_labels.get(size_nm, int(size_nm))
        ls = "-" if size_nm == 20.0 else "--"
        axes[0].plot(g["threshold"], g["D_mean"], marker=marker, linestyle=ls, color=color_dark,
                     label=f"{condition}, {label_nm} nm")
        axes[1].plot(g["threshold"], g["n_mean"], marker=marker, linestyle=ls, color=color_dark,
                     label=f"{condition}, {label_nm} nm")
        for x, y, n in zip(g["threshold"], g["D_mean"], g["n_tracks"]):
            if np.isfinite(y):
                axes[0].annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), fontsize=6)

    axes[0].set_xlabel("Minimum track length (frames)")
    axes[0].set_ylabel(r"Ensemble $D$ (µm²/s), approx.")
    axes[1].set_xlabel("Minimum track length (frames)")
    axes[1].set_ylabel(r"Ensemble $n$, approx.")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Min-track-length sensitivity (mean of per-track fits above threshold -- "
                 "approximation, see script docstring)", fontsize=10, fontweight="semibold")
    return fig


def main() -> None:
    df_water = _load_pickle(CACHE_D0, "TaskB1_TrackLengthBias_Compute.py")
    df_gel = _load_pickle(CACHE_20MG, "TaskB1_TrackLengthBias_Compute.py")

    table = pd.concat([
        sensitivity_table(df_water, "water"),
        sensitivity_table(df_gel, "hydrogel"),
    ], ignore_index=True)
    print(table.to_string(index=False))

    dls_labels = get_dls_labels()
    with plt.rc_context(_RC):
        fig = plot_sensitivity(table, dls_labels)
        plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(SAVE_PATH / "min_track_length_sensitivity.png", dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {SAVE_PATH / 'min_track_length_sensitivity.png'}")

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = SAVE_PATH / "min_track_length_sensitivity_table.csv"
        table.to_csv(csv_path, index=False)
        print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
