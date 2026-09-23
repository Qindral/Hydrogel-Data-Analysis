"""
Refined re-plot of eMSD_files_histogram.py's four per-file (one point = one
movie) metric grids, per item 4 of the refined-figures spec: Counts instead
of Density, identical bin edges and Counts y-limit shared across all size
panels within a metric, exponent n fixed to [-0.3, 1.2], D/Step Size on a
shared data-derived range, outer-only axis labels, bolder panel titles,
English text, DLS-labeled sizes.

Follow-up refinements from user feedback on the first version:
  - Track length and step size are now the per-file MEDIAN (not mean) --
    build_file_table() in eMSD_files_histogram.py computes the mean and
    that script must not be modified, so this script has its own
    build_file_table_median() instead. It re-derives both values directly
    from cache/msd_d0_results.pkl / cache/msd_20mg_files.pkl the same way
    the original does (core.analysis.calculate_step_sizes for step size,
    tracks_df.groupby("particle").size() for track length), just taking
    the median instead of the mean -- D/n are untouched, still read
    straight from fit_results_MSD as before.
  - Track length axis range changed from [0, 200] to [0, 50] frames: a
    per-file MEDIAN track length clusters far below the per-track range
    that [0, 200] was originally chosen for (see iMSD_histograms_Refined.py,
    where the axis is genuinely per-track).
  - The out-of-range footer annotation was removed from the figure itself
    (still reported to the console) per user request.
  - Every metric now also gets an additional "_kde" output with a
    Counts-rescaled KDE curve overlaid (_size_grid_utils.plot_count_grid_
    by_size(..., overlay_kde=True)), alongside the plain Counts-only file.

Does NOT modify eMSD_files_histogram.py or any cache -- pure consumer.
D/n columns and the file list itself still come from build_file_table()
(unmodified import); only the track-length/step-size aggregation is
re-derived locally as described above, from the same unmodified caches.

Outputs (own folder, PNG, 600 dpi): emsd_n_by_size(.png / _kde.png),
emsd_D_by_size(.png / _kde.png), emsd_stepsize_by_size(.png / _kde.png),
emsd_tracklength_by_size(.png / _kde.png).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.core.analysis import calculate_step_sizes
from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.eMSD_files_histogram import (
    build_file_table, CACHE_MSD_D0, CACHE_MSD_20MG, _load_pickle,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined._size_grid_utils import (
    compute_axis_spec, plot_count_grid_by_size,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Figures_Refined" / "eMSD_Histograms"

STYLE_WATER = dict(face="#009900", edge="#006600", label="Water")
STYLE_HYDROGEL = dict(face="#0000da", edge="#000099", label="Hydrogel")

_METRICS = [
    ("exponent", r"Anomalous exponent $n$", False, (-0.3, 1.2), "emsd_n_by_size"),
    ("D_um2_per_s", r"$D$ ($\mathrm{\mu m^2/s}$)", True, None, "emsd_D_by_size"),
    ("step_um_median", "Median step size per file (µm)", False, None, "emsd_stepsize_by_size"),
    ("track_length_frames_median", "Median track length per file (frames)", False, (0.0, 50.0),
     "emsd_tracklength_by_size"),
]


def build_file_table_median() -> pd.DataFrame:
    """Same per-file rows as eMSD_files_histogram.py::build_file_table(),
    but track length and step size are the per-file MEDIAN instead of the
    MEAN (D/n unchanged, straight from fit_results_MSD)."""
    rows = []
    sources = (("water", CACHE_MSD_D0), ("hydrogel", CACHE_MSD_20MG))
    for condition, cache_path in sources:
        results = _load_pickle(cache_path, "MSD_FromTrackmate_D0.py / _20mg.py")
        for r in results.values():
            tracks_df = r.get("tracks_df")
            mpp = r.get("mpp")
            fit = r.get("fit_results_MSD")
            if tracks_df is None or tracks_df.empty or mpp is None or fit is None:
                continue
            lengths = tracks_df.groupby("particle").size()
            steps = calculate_step_sizes(tracks_df, step_interval=1, sliding=True)
            step_um_median = (
                float(np.median(np.hypot(steps["dx"].to_numpy(), steps["dy"].to_numpy()) * mpp))
                if not steps.empty else np.nan
            )
            rows.append({
                "xml_path": r.get("xml_path"), "base_name": r.get("base_name"),
                "particle_size_nm": r.get("particle_size_nm"), "condition": condition,
                "track_length_frames_median": float(lengths.median()) if len(lengths) else np.nan,
                "step_um_median": step_um_median,
                "D_um2_per_s": fit.get("D_um2_per_s"), "exponent": fit.get("exponent"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = build_file_table()
    n_water = int((df["condition"] == "water").sum())
    n_hydrogel = int((df["condition"] == "hydrogel").sum())
    print(f"Per-file table (D/n): {len(df)} files ({n_water} water, {n_hydrogel} hydrogel)")

    df_median = build_file_table_median()
    print(f"Per-file table (median track length/step size): {len(df_median)} files")

    dls_labels = get_dls_labels()
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    tables = {
        "exponent": df, "D_um2_per_s": df,
        "step_um_median": df_median, "track_length_frames_median": df_median,
    }
    with plt.rc_context(_RC):
        for metric, xlabel, log_x, x_range, stem in _METRICS:
            table = tables[metric]
            spec, n_out = compute_axis_spec(table, metric, log_x=log_x, x_range=x_range)
            if n_out:
                print(f"  [NOTE] {metric}: {n_out} file(s) outside displayed range {x_range} "
                      "-- counted, not shown, not dropped from the underlying data.")
            for overlay_kde, suffix in ((False, ""), (True, "_kde")):
                fig = plot_count_grid_by_size(table, metric, xlabel, dls_labels, spec,
                                               style_water=STYLE_WATER, style_hydrogel=STYLE_HYDROGEL,
                                               overlay_kde=overlay_kde)
                fig_path = SAVE_PATH / f"{stem}{suffix}.png"
                safe_savefig(fig, fig_path, dpi=600, bbox_inches="tight")
                plt.close(fig)
                print(f"Plot gespeichert: {fig_path}")


if __name__ == "__main__":
    main()
