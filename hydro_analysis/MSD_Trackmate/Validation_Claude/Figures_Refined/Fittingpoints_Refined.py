"""
Refined re-plot of Fittingpoints_robustness.py's D/n-vs-fit-points robustness
check, per item 6 of the refined-figures spec: one separate 6-panel figure
each for D and n (panels = particle size, not lines-in-one-axes), one
version using all available fit-point counts (4-8) and one using only
4/5/6, linear y-axes for both D and n, no dashed/dotted lines for the two
hydrogel conditions -- markers distinguish condition instead (solid lines
throughout). Size colors use the same orange (small) -> blue (large)
palette already established for size in Correlations.py. English labels,
DLS size labels, shared axes with outer-only labels, bolder panel titles.

Does NOT modify Fittingpoints_robustness.py or any cache. Dissertation
version (Abb14, Abb15; user, 2026-10-14): hydrogel surface loading and
injection are shown as one "Hydrogel" curve against buffer. The per-condition
summary in fittingpoints_robustness.csv cannot be merged, so the per-track
fits are rebuilt here with that script's own, unchanged functions
(per_track_fits_by_points on cache/msd_d0_results.pkl and
cache/msd_20mg_files.pkl, then build_summary: median and IQR over all tracks
of a size x condition x fit-point count, no weighting). Only the condition
label is merged before build_summary. As a check, the unmerged summary is
compared with fittingpoints_robustness.csv and the largest deviation printed.

Colours: SIZE_COLORS; buffer markers and band in the base colour, hydrogel in
the dark colour; lines in the dark colour. Abb14: D on a log axis so that
values below 5 um^2 s^-1 are readable. Abb15: n linear as before.

Also reports the median and mean track length t_l = f_max - f_min per size and
medium for tracks with >= MIN_TRACK_FRAMES detections (core.analysis.
MIN_TRACK_LENGTH) from the tracks_df of both caches, and writes it to
fittingpoints_tracklength_summary.csv.

Dissertation outputs (PDF, SAVE_PATH): fittingpoints_D_by_size_456_only.pdf,
fittingpoints_n_by_size_456_only.pdf, fittingpoints_robustness_hydrogel_combined.csv,
fittingpoints_tracklength_summary.csv. The all-points variants and the
200-1370 nm D variant go to SCRATCH_PATH (PNG).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from hydro_analysis.core.analysis import MIN_TRACK_LENGTH
from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS
from hydro_analysis.MSD_Trackmate.Validation_Claude.Fittingpoints_robustness import (
    CACHE_D0, CACHE_20MG, _load_pickle, per_track_fits_by_points, build_summary,
)

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
# per-condition summary written by Fittingpoints_robustness.py (used only as a check)
CSV_PATH = RESULTS / "_scratch" / "Auswertungsbilder" / "Fittingpoints_Robustness" / "fittingpoints_robustness.csv"
SAVE_PATH = RESULTS / "SPT"
SCRATCH_PATH = RESULTS / "_scratch" / "Fittingpoints_Refined"
STYLE = Path(__file__).resolve().parents[3] / "thesis.mplstyle"
WIDTH_FULL_IN = 6.30
MIN_TRACK_FRAMES = MIN_TRACK_LENGTH   # approved minimum track length (10 frames)

# hydrogel surface loading and injection merged into one curve (user, 2026-10-14)
MERGE_CONDITIONS = {"Hydrogel surface loading": "Hydrogel", "Hydrogel injection": "Hydrogel"}
CONDITION_MARKERS = {"Wasser": "o", "Hydrogel": "s"}
CONDITION_LABELS_EN = {"Wasser": "Buffer", "Hydrogel": "Hydrogel"}
D_LOG_RANGE = (0.1, 40.0)   # Abb14 log axis, covers all medians and IQR bands; PROPOSED, not approved


def plot_metric_by_size(df: pd.DataFrame, value_col: str, iqr_low_col: str, iqr_high_col: str,
                         ylabel: str, dls_labels: dict[float, int], fit_points: list[int],
                         sizes_filter: set[float] | None = None,
                         y_range: tuple[float, float] | None = None, log_y: bool = False) -> plt.Figure:
    sizes = [s for s in sorted(df["partikelgröße_nm"].unique())
             if s in SIZE_COLORS and (sizes_filter is None or s in sizes_filter)]
    n = len(sizes)
    ncols = 3 if n > 3 else max(n, 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_FULL_IN, 1.75 * nrows + 0.55),
                              squeeze=False, sharex=True, sharey=True)
    ax_flat = axes.flatten()
    for ax in ax_flat[n:]:
        ax.set_visible(False)

    legend_conditions: dict[str, str] = {}
    for ax, size_nm in zip(ax_flat, sizes):
        face, edge = SIZE_COLORS[size_nm]
        sub_size = df[df["partikelgröße_nm"] == size_nm]
        for condition, g in sub_size.groupby("bedingung"):
            g = g.sort_values("fittingpoints")
            x = g["fittingpoints"].to_numpy()
            y = g[value_col].to_numpy()
            marker = CONDITION_MARKERS.get(condition, "o")
            fill = face if condition == "Wasser" else edge   # buffer base, hydrogel dark
            ax.plot(x, y, color=edge, linewidth=1.2, linestyle="-", marker=marker,
                     markersize=4, markerfacecolor=fill, markeredgecolor=edge, markeredgewidth=0.6)
            ax.fill_between(x, g[iqr_low_col], g[iqr_high_col], color=fill, alpha=0.18, linewidth=0)
            legend_conditions.setdefault(condition, marker)
        ax.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm")
        ax.set_xticks(fit_points)
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        if y_range is not None:
            ax.set_ylim(*y_range)
        ax.label_outer()

    for ax in ax_flat[:n]:
        ax.set_xlabel("Number of fit points")
    for ax in ax_flat[:n]:
        ax.label_outer()
    fig.supylabel(ylabel, fontsize=9)   # one label for both rows

    handles = [Line2D([0], [0], color="0.3", marker=marker, linestyle="-", markersize=4,
                       markerfacecolor=("0.75" if cond == "Wasser" else "0.3"), markeredgecolor="0.3",
                       label=CONDITION_LABELS_EN.get(cond, cond))
               for cond, marker in sorted(legend_conditions.items(), key=lambda kv: kv[0] != "Wasser")]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="0.6", alpha=0.3, label="Interquartile range of tracks"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    return fig


def tracklength_summary(results: dict, medium: str) -> pd.DataFrame:
    """Median and mean t_l = f_max - f_min (frames, gaps included) per particle size for
    tracks with >= MIN_TRACK_FRAMES detections, from the cached tracks_df."""
    rows = []
    lengths: dict[float, list[np.ndarray]] = {}
    for r in results.values():
        tracks = r.get("tracks_df")
        size_nm = r.get("particle_size_nm")
        if tracks is None or tracks.empty or size_nm is None:
            continue
        g = tracks.groupby("particle")["frame"]
        n_det = g.size()
        t_l = (g.max() - g.min())[n_det >= MIN_TRACK_FRAMES]
        lengths.setdefault(float(size_nm), []).append(t_l.to_numpy(dtype=float))
    for size_nm, parts in sorted(lengths.items()):
        t = np.concatenate(parts)
        rows.append({"medium": medium, "particle_size_nm": size_nm, "n_files": len(parts), "n_tracks": int(t.size),
                     "median_t_l_frames": float(np.median(t)) if t.size else np.nan,
                     "mean_t_l_frames": float(np.mean(t)) if t.size else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    d0_results = _load_pickle(CACHE_D0, "MSD_FromTrackmate_D0.py")
    hydrogel_results = _load_pickle(CACHE_20MG, "MSD_FromTrackmate_20mg.py")
    per_track = pd.concat([per_track_fits_by_points(d0_results, "water"),
                           per_track_fits_by_points(hydrogel_results, "hydrogel")], ignore_index=True)

    if CSV_PATH.exists():
        reference = pd.read_csv(CSV_PATH)
        check = build_summary(per_track).merge(reference, on=["partikelgröße_nm", "bedingung", "fittingpoints"],
                                               suffixes=("", "_csv"))
        dev = max(float(np.nanmax(np.abs(check[c] - check[f"{c}_csv"]))) for c in ("D_median", "n_median"))
        print(f"Check against {CSV_PATH.name}: {len(check)} rows matched, max |deviation| of medians = {dev:.3g}")
    else:
        print(f"[NOTE] {CSV_PATH} not found -- consistency check skipped.")

    per_track["condition"] = per_track["condition"].replace(MERGE_CONDITIONS)
    df = build_summary(per_track)
    dls_labels = get_dls_labels()

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAVE_PATH / "fittingpoints_robustness_hydrogel_combined.csv", index=False)
    print(df[df["fittingpoints"].isin((4, 5, 6))].to_string(index=False))

    tl = pd.concat([tracklength_summary(d0_results, "Buffer"),
                    tracklength_summary(hydrogel_results, "Hydrogel")], ignore_index=True)
    tl.insert(2, "label_nm", tl["particle_size_nm"].map(lambda s: dls_labels.get(s, int(s))))
    tl.to_csv(SAVE_PATH / "fittingpoints_tracklength_summary.csv", index=False)
    print(f"Track length t_l = f_max - f_min, tracks >= {MIN_TRACK_FRAMES} detections:")
    print(tl.to_string(index=False))

    d_label = r"Diffusion coefficient $D$ (µm$^2$ s$^{-1}$)"
    n_label = r"Anomalous exponent $n$"
    fit_points_456 = [p for p in (4, 5, 6) if p in df["fittingpoints"].unique()]
    df_456 = df[df["fittingpoints"].isin(fit_points_456)]
    with plt.style.context(STYLE):
        fig_d = plot_metric_by_size(df_456, "D_median", "D_IQR_low", "D_IQR_high", d_label, dls_labels,
                                    fit_points_456, y_range=D_LOG_RANGE, log_y=True)
        fig_d.savefig(SAVE_PATH / "fittingpoints_D_by_size_456_only.pdf")
        plt.close(fig_d)
        fig_n = plot_metric_by_size(df_456, "n_median", "n_IQR_low", "n_IQR_high", n_label, dls_labels,
                                    fit_points_456)
        fig_n.savefig(SAVE_PATH / "fittingpoints_n_by_size_456_only.pdf")
        plt.close(fig_n)
        print(f"Plots gespeichert: {SAVE_PATH}")

    # exploratory variants, not used in the dissertation
    with plt.rc_context(_RC):
        all_points = sorted(df["fittingpoints"].unique())
        for value, lo, hi, label, stem in (("D_median", "D_IQR_low", "D_IQR_high", d_label, "D"),
                                           ("n_median", "n_IQR_low", "n_IQR_high", n_label, "n")):
            fig = plot_metric_by_size(df, value, lo, hi, label, dls_labels, all_points)
            safe_savefig(fig, SCRATCH_PATH / f"fittingpoints_{stem}_by_size_all_points.png", dpi=600)
            plt.close(fig)
        fig = plot_metric_by_size(df_456, "D_median", "D_IQR_low", "D_IQR_high", d_label, dls_labels,
                                  fit_points_456, sizes_filter={200.0, 500.0, 1000.0}, y_range=(0.0, 4.0))
        safe_savefig(fig, SCRATCH_PATH / "fittingpoints_D_by_size_456_only_200plus_0to4.png", dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    main()
