"""Refined SNR-by-size boxplots with DLS labels and condition colors.

Consumer script, two inputs, no raw data:
  - cache/particle_size_snr.pkl (written by Particle_Size_SNR_Compute.py --mode all),
    table "tracks" -- one SNR value per track (median over its detections), all
    tracked files per condition and size. SNR = (peak within R - annulus mean) /
    annulus std, as in SNR_Analysis.py. -> Abb21 snr_overview_by_size_without_immobilized.
  - PAIRED_CSV, snr_dog_particle_metrics.csv written by snr_signal_profile.py:
    a small set of particles measured before (raw) and after DoG filtering in the
    same frame, 0-5 particles per condition and size. -> Abb22
    snr_raw_vs_dog_by_size_without_immobilized. Because Abb22 uses only this paired
    subset and Abb21 all tracks, the buffer values of the two figures differ; the
    group sizes are printed, not drawn.

Dissertation figures (user, 2026-10-14; Styleguide v2, width class narrow, PDF,
SAVE_PATH): no zero line, "Buffer" instead of "Water", SIZE_COLORS with buffer in the
base and hydrogel in the dark colour; in Abb22 raw boxes filled, DoG boxes hatched,
no title and no group counts in the figure. The variant with immobilized particles
and the paired SNR change plot are not used and go to SCRATCH_PATH (PNG).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS
from hydro_analysis.MSD_Trackmate.Validation_Claude.SNR_Analysis import PSS_CACHE, load_track_snr

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
SAVE_PATH = RESULTS / "SPT"
SCRATCH_PATH = RESULTS / "_scratch" / "SNR_Refined"
PAIRED_CSV = SAVE_PATH / "snr_dog_particle_metrics.csv"
STYLE = Path(__file__).resolve().parents[3] / "thesis.mplstyle"
WIDTH_NARROW_IN, RATIO = 4.72, 1.42
CONDITION_ORDER = ("water", "hydrogel", "immobilized")
CONDITION_LABELS = {"water": "Buffer", "hydrogel": "Hydrogel", "immobilized": "Immobilized"}
LEGEND_SIZES_NM = (20.0, 50.0)      # sizes measured in both media, shown in the legend swatches


def condition_colors(size_nm: float, condition: str) -> tuple[str, str]:
    """(face, edge): buffer base, hydrogel dark colour of the particle size; immobilized grey."""
    if condition == "immobilized":
        return "#bdbdbd", "#4d4d4d"
    base, dark = SIZE_COLORS[size_nm]
    return (dark if condition == "hydrogel" else base), dark


def _ordered(conditions) -> list[str]:
    return [c for c in CONDITION_ORDER if c in set(conditions)]


def _condition_handles(conditions: list[str]) -> tuple[list, list[str]]:
    handles, labels = [], []
    for c in conditions:
        if c == "immobilized":
            handles.append(Patch(facecolor="#bdbdbd", edgecolor="#4d4d4d"))
        else:
            handles.append(tuple(Patch(facecolor=condition_colors(s, c)[0], edgecolor=condition_colors(s, c)[1])
                                 for s in LEGEND_SIZES_NM))
        labels.append(CONDITION_LABELS.get(c, c))
    return handles, labels


def _style_box(bp, face: str, edge: str, alpha: float, hatch: str | None = None) -> None:
    for patch in bp["boxes"]:
        patch.set_facecolor(face)
        patch.set_edgecolor(edge)
        patch.set_alpha(alpha)
        patch.set_linewidth(0.8)
        if hatch:
            patch.set_hatch(hatch)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(edge)
            line.set_linewidth(0.8)
    for line in bp["medians"]:
        line.set_color("black")
        line.set_linewidth(1.0)


def plot_snr_overview(df: pd.DataFrame, dls_labels: dict[float, int], conditions: list[str],
                      figsize=(WIDTH_NARROW_IN, WIDTH_NARROW_IN / RATIO)) -> plt.Figure:
    sizes = sorted(df["particle_size_nm"].unique())
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(sizes))
    width = 0.8 / max(len(conditions), 1)
    for i, condition in enumerate(conditions):
        offsets = positions + (i - (len(conditions) - 1) / 2) * width
        for x, size in zip(offsets, sizes):
            values = df.loc[(df["particle_size_nm"] == size) & (df["condition"] == condition), "snr"].to_numpy()
            if not len(values):
                continue
            bp = ax.boxplot([values], positions=[x], widths=width * 0.85, patch_artist=True, showfliers=False)
            _style_box(bp, *condition_colors(size, condition), alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(dls_labels.get(s, int(s))) for s in sizes])
    ax.set_xlim(-0.6, len(sizes) - 0.4)
    ax.set_xlabel("Particle size (nm)")
    ax.set_ylabel("Signal-to-noise ratio")
    handles, labels = _condition_handles(conditions)
    ax.legend(handles, labels, loc="upper left", handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)})
    return fig


def load_paired_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Nicht gefunden: {path}\nBitte zuerst snr_signal_profile.py ausführen.")
    df = pd.read_csv(path)
    required = {"condition", "particle_size_nm", "snr", "dog_snr", "frame", "spot_id"}
    if missing := required - set(df.columns):
        raise ValueError(f"Paired SNR input missing columns: {sorted(missing)}")
    keys = ["condition", "particle_size_nm", "frame", "spot_id"]
    if df.duplicated(keys).any():
        raise ValueError("Paired SNR input contains duplicate particle/frame measurements")
    valid = np.isfinite(df["snr"]) & np.isfinite(df["dog_snr"])
    if not valid.all():
        print(f"Excluded {(~valid).sum()} non-finite SNR pairs")
    df = df.loc[valid].copy()
    df["snr_change_percent"] = np.where(df["snr"] > 0, 100 * (df["dog_snr"] / df["snr"] - 1), np.nan)
    return df


def plot_paired_overview(df: pd.DataFrame, dls_labels: dict[float, int], conditions: list[str], *,
                         change: bool = False, figsize=(WIDTH_NARROW_IN, WIDTH_NARROW_IN / RATIO)) -> plt.Figure:
    """Condition-grouped boxes; adjacent raw (filled) / DoG (hatched) boxes share paired samples."""
    sizes = sorted(df["particle_size_nm"].unique())
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(sizes))
    width = 0.8 / max(len(conditions), 1)
    stages = [("snr_change_percent", None)] if change else [("snr", None), ("dog_snr", "////")]
    for i, condition in enumerate(conditions):
        for j, size in enumerate(sizes):
            group = df[(df["particle_size_nm"] == size) & (df["condition"] == condition)]
            face, edge = condition_colors(size, condition)
            center = positions[j] + (i - (len(conditions) - 1) / 2) * width
            for stage_idx, (column, hatch) in enumerate(stages):
                values = group[column].to_numpy()
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                position = center + (stage_idx - (len(stages) - 1) / 2) * width / len(stages)
                bp = ax.boxplot([values], positions=[position], widths=width * 0.8 / len(stages),
                                patch_artist=True, showfliers=False)
                _style_box(bp, face if hatch is None else "white", edge, alpha=0.85, hatch=hatch)
                if len(values) == 1:
                    ax.scatter([position], values, s=12, facecolor=face, edgecolor=edge, linewidth=0.6, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(dls_labels.get(s, int(s))) for s in sizes])
    ax.set_xlim(-0.6, len(sizes) - 0.4)
    ax.set_xlabel("Particle size (nm)")
    ax.set_ylabel("SNR change after DoG (%)" if change else "Signal-to-noise ratio")
    handles, labels = _condition_handles(conditions)
    if not change:
        handles += [Patch(facecolor="0.6", edgecolor="0.3"), Patch(facecolor="white", edgecolor="0.3", hatch="////")]
        labels += ["Raw", "DoG"]
    ax.legend(handles, labels, loc="upper left", handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)})
    return fig


def main(pss_cache=PSS_CACHE) -> None:
    df = load_track_snr(Path(pss_cache))
    paired = load_paired_data(PAIRED_CSV)
    dls_labels = get_dls_labels()
    all_conditions = _ordered(df["condition"].unique())
    without = [c for c in all_conditions if c != "immobilized"]
    print("Abb21 (all tracks, particle_size_snr.pkl):")
    print(df.groupby(["condition", "particle_size_nm"]).agg(n_tracks=("snr", "size"), n_files=("file", "nunique"),
                                                            median_snr=("snr", "median")).to_string())
    print("Abb22 (paired subset, snr_dog_particle_metrics.csv):")
    print(paired.groupby(["condition", "particle_size_nm"]).agg(n=("snr", "size"), raw_median=("snr", "median"),
                                                                dog_median=("dog_snr", "median")).to_string())

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
    paired_without = paired[paired["condition"] != "immobilized"]
    with plt.style.context(STYLE):
        fig = plot_snr_overview(df[df["condition"].isin(without)], dls_labels, without)
        fig.savefig(SAVE_PATH / "snr_overview_by_size_without_immobilized.pdf")
        plt.close(fig)
        fig = plot_paired_overview(paired_without, dls_labels, _ordered(paired_without["condition"].unique()))
        fig.savefig(SAVE_PATH / "snr_raw_vs_dog_by_size_without_immobilized.pdf")
        plt.close(fig)
        print(f"Plots gespeichert: {SAVE_PATH}")

    # exploratory variants, not used in the dissertation
    with plt.rc_context(_RC):
        big = (7.15, 5.00)
        if "immobilized" in all_conditions:
            fig = plot_snr_overview(df, dls_labels, all_conditions, figsize=big)
            safe_savefig(fig, SCRATCH_PATH / "snr_overview_by_size_with_immobilized.png", dpi=600)
            plt.close(fig)
        for subset, suffix in ((paired, "with_immobilized"), (paired_without, "without_immobilized")):
            conditions = _ordered(subset["condition"].unique())
            if suffix == "with_immobilized":
                fig = plot_paired_overview(subset, dls_labels, conditions, figsize=big)
                safe_savefig(fig, SCRATCH_PATH / f"snr_raw_vs_dog_by_size_{suffix}.png", dpi=600)
                plt.close(fig)
            fig = plot_paired_overview(subset, dls_labels, conditions, change=True, figsize=big)
            safe_savefig(fig, SCRATCH_PATH / f"snr_dog_change_by_size_{suffix}.png", dpi=600)
            plt.close(fig)


if __name__ == "__main__":
    main()
