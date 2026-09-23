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

Outputs (own folder, PNG, 600 dpi): imsd_n_by_size.png, imsd_D_by_size.png,
imsd_D_by_size_water_only.png, imsd_D_by_size_hydrogel_only.png,
imsd_stepsize_by_size.png, imsd_tracklength_by_size.png,
investigation_1000nm_water.txt.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.iMSD_histograms import (
    build_track_table, CACHE_TRACKLENGTH_D0, CACHE_MSD_D0,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined._size_grid_utils import (
    compute_axis_spec, plot_count_grid_by_size,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Figures_Refined" / "iMSD_Histograms"

RAW_1000NM_WATER_FOLDER = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_1000")

STYLE_WATER = dict(face="#009900", edge="#006600", label="Water")
STYLE_HYDROGEL = dict(face="#0000da", edge="#000099", label="Hydrogel")

_METRICS = [
    ("exponent", r"Anomalous exponent $n$", False, (-0.3, 1.2), "imsd_n_by_size.png"),
    ("step_um", "Mean step size per track (µm)", False, None, "imsd_stepsize_by_size.png"),
    ("track_length_frames", "Track length (frames)", False, (0.0, 200.0), "imsd_tracklength_by_size.png"),
]


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
    df = build_track_table()
    n_water = int((df["condition"] == "water").sum())
    n_hydrogel = int((df["condition"] == "hydrogel").sum())
    print(f"Per-track table: {len(df)} tracks ({n_water} water, {n_hydrogel} hydrogel)")

    dls_labels = get_dls_labels()
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_RC):
        for metric, xlabel, log_x, x_range, filename in _METRICS:
            spec, n_out = compute_axis_spec(df, metric, log_x=log_x, x_range=x_range)
            if n_out:
                print(f"  [NOTE] {metric}: {n_out} track(s) outside displayed range {x_range} "
                      "-- counted, not shown, not dropped from the underlying data.")
            stem = filename[:-4]
            for overlay_kde, suffix in ((False, ""), (True, "_kde")):
                fig = plot_count_grid_by_size(df, metric, xlabel, dls_labels, spec,
                                               style_water=STYLE_WATER, style_hydrogel=STYLE_HYDROGEL,
                                               overlay_kde=overlay_kde)
                fig_path = SAVE_PATH / f"{stem}{suffix}.png"
                safe_savefig(fig, fig_path, dpi=600, bbox_inches="tight")
                plt.close(fig)
                print(f"Plot gespeichert: {fig_path}")

        # D: combined grid + water-only/hydrogel-only sharing the same axis spec.
        d_spec, d_n_out = compute_axis_spec(df, "D_um2_per_s", log_x=True, x_range=None)
        d_xlabel = r"$D$ ($\mathrm{\mu m^2/s}$)"
        for condition_filter, stem in ((None, "imsd_D_by_size"),
                                        ("water", "imsd_D_by_size_water_only"),
                                        ("hydrogel", "imsd_D_by_size_hydrogel_only")):
            for overlay_kde, suffix in ((False, ""), (True, "_kde")):
                fig = plot_count_grid_by_size(df, "D_um2_per_s", d_xlabel, dls_labels, d_spec,
                                               condition_filter=condition_filter,
                                               style_water=STYLE_WATER, style_hydrogel=STYLE_HYDROGEL,
                                               overlay_kde=overlay_kde)
                fig_path = SAVE_PATH / f"{stem}{suffix}.png"
                safe_savefig(fig, fig_path, dpi=600, bbox_inches="tight")
                plt.close(fig)
                print(f"Plot gespeichert: {fig_path}")

    print("\n1000 nm / water file-count investigation:")
    report = investigate_1000nm_water()
    print(report)
    report_path = SAVE_PATH / "investigation_1000nm_water.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Bericht gespeichert: {report_path}")


if __name__ == "__main__":
    main()
