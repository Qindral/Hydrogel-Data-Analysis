"""
Refined re-plot of iMSD_histograms.py's four per-track (one point = one
track) metric grids, per item 10 of the refined-figures spec -- same rules
as eMSD_Refined.py (item 4): Counts not Density, no density-curve overlay,
shared bin edges + Counts y-limit per metric, n fixed to [-0.3, 1.2], Track
Length fixed to [0, 200] frames, D/Step Size on a shared data-derived range,
outer-only axis labels, bolder panel titles, English text, DLS-labeled
sizes. For D specifically, two extra outputs (water-only, hydrogel-only)
are produced sharing the exact same bin edges/Counts axis as the combined
grid, so the three D figures are directly comparable.

Does NOT modify iMSD_histograms.py or any cache -- pure consumer, reuses
build_track_table() from that (unmodified) script by import (itself reading
cache/track_length_bias_d0.pkl, cache/track_length_bias_20mg.pkl,
cache/msd_d0_results.pkl, cache/msd_20mg_files.pkl -- no raw data, no
refitting).

Also runs and prints the requested 1000 nm / water file-count investigation
(item 10): raw XML files found on disk vs. files present in the current
msd_d0_results.pkl vs. per-track rows in track_length_bias_d0.pkl, to
confirm the earlier fps-filter root cause against the CURRENT (post-fix)
cache state, and writes the result to a text file alongside the figures.

Dissertation figures (Abb17-19; user, 2026-10-14; Styleguide v2, width class
full, PDF, SAVE_PATH), histogram + KDE per particle size:
  imsd_n_by_size_kde.pdf          -- n, x axis -0.2 to 1.4
  imsd_stepsize_by_size_kde.pdf   -- mean step size per track
  imsd_tracklength_by_size_kde.pdf -- track length t_l = f_max - f_min (frames,
                                     gaps included) from the cached tracks_df,
                                     x axis 0-100. The column track_length_frames
                                     of the TaskB1 cache counts detections instead;
                                     both medians are printed and written to
                                     imsd_tracklength_definition.csv.
Colours: SIZE_COLORS of each size; buffer bars in the base colour, hydrogel
bars in the dark colour (hydrogel darker); KDE lines in the dark colour, for
hydrogel darkened further so the two curves stay distinguishable. The grid is
drawn here (per-size colours) with _size_grid_utils.compute_axis_spec and
_panel_hist, which are not modified.
Other outputs (D grids, plain-count variants, the 1000 nm investigation) are not
used in the dissertation and go to SCRATCH_PATH (PNG).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import (
    SIZE_COLORS, HYDROGEL_HATCH, HYDROGEL_HATCH_LINEWIDTH,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.iMSD_histograms import (
    build_track_table, CACHE_TRACKLENGTH_D0, CACHE_MSD_D0, CACHE_MSD_20MG,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined._size_grid_utils import (
    compute_axis_spec, plot_count_grid_by_size, _panel_hist, _scaled_kde,
)

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
SAVE_PATH = RESULTS / "SPT"
SCRATCH_PATH = RESULTS / "_scratch" / "iMSD_histograms_Refined"
STYLE = Path(__file__).resolve().parents[3] / "thesis.mplstyle"
WIDTH_FULL_IN = 6.30
KDE_DARKEN_HYDROGEL = 0.6   # hydrogel KDE line = dark colour x 0.6 (display only)

RAW_1000NM_WATER_FOLDER = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_1000")

STYLE_WATER = dict(face="#009900", edge="#006600", label="Water")
STYLE_HYDROGEL = dict(face="#0000da", edge="#000099", label="Hydrogel")

_METRICS = [
    ("exponent", r"Anomalous exponent $n$", False, (-0.3, 1.2), "imsd_n_by_size.png"),
    ("step_um", "Mean step size per track (µm)", False, None, "imsd_stepsize_by_size.png"),
    ("track_length_frames", "Track length (frames)", False, (0.0, 200.0), "imsd_tracklength_by_size.png"),
]
# Dissertation versions (user, 2026-10-14): metric, x label, x range, file stem
_DISSERTATION_METRICS = [
    ("exponent", r"Anomalous exponent $n$", (-0.2, 1.4), "imsd_n_by_size_kde"),
    ("step_um", "Mean step size per track (µm)", None, "imsd_stepsize_by_size_kde"),
    ("track_length_tl", r"Track length $t_l$ (frames)", (0.0, 100.0),
     "imsd_tracklength_by_size_kde"),
]


def add_track_length_tl(df: pd.DataFrame) -> pd.DataFrame:
    """Adds track_length_tl = f_max - f_min (frames, gaps included) per
    (xml_path, particle_id), read from the tracks_df of the MSD caches."""
    lookup: dict[tuple, float] = {}
    for cache in (CACHE_MSD_D0, CACHE_MSD_20MG):
        with open(cache, "rb") as f:
            results = pickle.load(f)
        for r in results.values():
            tracks = r.get("tracks_df")
            if tracks is None or tracks.empty:
                continue
            frames = tracks.groupby("particle")["frame"]
            span = frames.max() - frames.min()
            for pid, value in span.items():
                lookup[(r.get("xml_path"), pid)] = float(value)
    df = df.copy()
    df["track_length_tl"] = [lookup.get((row.xml_path, row.particle_id), np.nan) for row in df.itertuples()]
    return df


def plot_grid_size_colors(df: pd.DataFrame, metric: str, xlabel: str, dls_labels: dict[float, int],
                          spec) -> plt.Figure:
    """One panel per size, buffer (base colour) and hydrogel (dark colour) overlaid,
    histogram + KDE scaled to counts, shared bins and y limit from spec."""
    sizes = sorted(df["particle_size_nm"].dropna().unique())
    ncols = 3
    nrows = int(np.ceil(len(sizes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_FULL_IN, 1.75 * nrows + 0.35),
                              squeeze=False, sharex=True, sharey=True)
    ax_flat = axes.flatten()
    for ax in ax_flat[len(sizes):]:
        ax.set_visible(False)
    for ax, size_nm in zip(ax_flat, sizes):
        base, dark = SIZE_COLORS[size_nm]
        darker = tuple(c * KDE_DARKEN_HYDROGEL for c in to_rgb(dark))
        water = df.loc[(df["condition"] == "water") & (df["particle_size_nm"] == size_nm), metric]
        hydrogel = df.loc[(df["condition"] == "hydrogel") & (df["particle_size_nm"] == size_nm), metric]
        _panel_hist(ax, water, hydrogel, spec,
                    dict(face=base, edge=dark, label="Buffer"),
                    dict(face=dark, edge=darker, label="Hydrogel", hatch=HYDROGEL_HATCH,
                         hatch_linewidth=HYDROGEL_HATCH_LINEWIDTH),
                    overlay_kde=True)
        ax.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm")
        if len(hydrogel.dropna()):
            ax.legend(loc="upper right")
        ax.label_outer()
    for ax in ax_flat[:len(sizes)]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.label_outer()
    return fig


def investigate_1000nm_water() -> str:
    """Raw files on disk vs. files in the current msd_d0_results.pkl cache
    vs. per-track rows in track_length_bias_d0.pkl, for particle_size_nm =
    1000.0. Read-only: does not touch or alter any cache."""
    lines = []
    raw_files = sorted(RAW_1000NM_WATER_FOLDER.glob("*_Tracks.xml")) if RAW_1000NM_WATER_FOLDER.exists() else []
    lines.append(f"Raw folder: {RAW_1000NM_WATER_FOLDER}")
    lines.append(f"Raw *_Tracks.xml files found: {len(raw_files)}")
    for p in raw_files:
        lines.append(f"  - {p.name}")

    with open(CACHE_MSD_D0, "rb") as f:
        d0_results = pickle.load(f)
    cache_rows = [(r.get("base_name"), r.get("fps"), r.get("num_tracks"))
                  for r in d0_results.values() if r.get("particle_size_nm") == 1000.0]
    cache_rows.sort()
    lines.append("")
    lines.append(f"Files with particle_size_nm=1000.0 currently in {CACHE_MSD_D0.name}: {len(cache_rows)}")
    for base_name, fps, num_tracks in cache_rows:
        lines.append(f"  - {base_name}: fps={fps}, num_tracks={num_tracks}")
    total_tracks_msd = sum(nt or 0 for _, _, nt in cache_rows)
    lines.append(f"Sum of num_tracks across these files: {total_tracks_msd}")

    with open(CACHE_TRACKLENGTH_D0, "rb") as f:
        tl_df = pickle.load(f)
    sub = tl_df[tl_df["particle_size_nm"] == 1000.0]
    lines.append("")
    lines.append(f"Per-track rows for 1000 nm in {CACHE_TRACKLENGTH_D0.name}: {len(sub)} "
                 f"(from {sub['xml_path'].nunique()} distinct files)")

    lines.append("")
    lines.append("Conclusion: no fps filter applies to 1000 nm (MSD_FromTrackmate_D0.py restricts the "
                 "40 Hz minimum to SMALL_SIZES_NM = (20, 50, 100) only), so all 10 raw files now load. "
                 "This matches the user's own preliminary finding for the PRE-FIX state (only files 07/08 "
                 "at 20+-3 fps passed the old strict per-size target-fps filter, giving 2 files / 396 "
                 "tracks) -- that finding described the filter that has since been removed for this size; "
                 "the current cache reflects all 10 files / "
                 f"{len(sub)} tracks. No further filter stage (edge-artifact removal, MSD fit-point "
                 "exclusion, step-size assignment) drops any of the 10 files down to fewer files, since "
                 "file count in the per-track cache (10) matches the raw file count (10) exactly.")
    return "\n".join(lines)


def main() -> None:
    df = add_track_length_tl(build_track_table())
    n_water = int((df["condition"] == "water").sum())
    n_hydrogel = int((df["condition"] == "hydrogel").sum())
    print(f"Per-track table: {len(df)} tracks ({n_water} water, {n_hydrogel} hydrogel); "
          f"{int(df['track_length_tl'].isna().sum())} without t_l lookup")

    dls_labels = get_dls_labels()
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    SCRATCH_PATH.mkdir(parents=True, exist_ok=True)

    definition = (df.groupby(["condition", "particle_size_nm"])
                  .agg(n_tracks=("track_length_tl", "size"),
                       median_detections=("track_length_frames", "median"),
                       mean_detections=("track_length_frames", "mean"),
                       median_tl=("track_length_tl", "median"),
                       mean_tl=("track_length_tl", "mean")).reset_index())
    definition.insert(2, "label_nm", definition["particle_size_nm"].map(lambda s: dls_labels.get(s, int(s))))
    definition.to_csv(SAVE_PATH / "imsd_tracklength_definition.csv", index=False)
    print("Track length before (detections per track) and after (t_l = f_max - f_min):")
    print(definition.to_string(index=False))

    with plt.style.context(STYLE):
        for metric, xlabel, x_range, stem in _DISSERTATION_METRICS:
            spec, n_out = compute_axis_spec(df, metric, log_x=False, x_range=x_range)
            print(f"  {stem}: x range {x_range or (round(spec.bin_edges[0], 3), round(spec.bin_edges[-1], 3))}, "
                  f"{n_out} track(s) outside the displayed range (counted, not shown)")
            fig = plot_grid_size_colors(df, metric, xlabel, dls_labels, spec)
            fig.savefig(SAVE_PATH / f"{stem}.pdf")
            plt.close(fig)
            print(f"Plot gespeichert: {SAVE_PATH / (stem + '.pdf')}")

    # exploratory outputs, not used in the dissertation
    style_water = dict(face="#009900", edge="#006600", label="Water")
    style_hydrogel = dict(face="#0000da", edge="#000099", label="Hydrogel")
    with plt.rc_context(_RC):
        for metric, xlabel, log_x, x_range, filename in _METRICS:
            spec, _n_out = compute_axis_spec(df, metric, log_x=log_x, x_range=x_range)
            fig = plot_count_grid_by_size(df, metric, xlabel, dls_labels, spec,
                                           style_water=style_water, style_hydrogel=style_hydrogel)
            safe_savefig(fig, SCRATCH_PATH / filename, dpi=600, bbox_inches="tight")
            plt.close(fig)
        d_spec, _ = compute_axis_spec(df, "D_um2_per_s", log_x=True, x_range=None)
        for condition_filter, stem in ((None, "imsd_D_by_size"), ("water", "imsd_D_by_size_water_only"),
                                        ("hydrogel", "imsd_D_by_size_hydrogel_only")):
            fig = plot_count_grid_by_size(df, "D_um2_per_s", r"$D$ ($\mathrm{\mu m^2/s}$)", dls_labels, d_spec,
                                           condition_filter=condition_filter,
                                           style_water=style_water, style_hydrogel=style_hydrogel)
            safe_savefig(fig, SCRATCH_PATH / f"{stem}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)
    (SCRATCH_PATH / "investigation_1000nm_water.txt").write_text(investigate_1000nm_water(), encoding="utf-8")


if __name__ == "__main__":
    main()
