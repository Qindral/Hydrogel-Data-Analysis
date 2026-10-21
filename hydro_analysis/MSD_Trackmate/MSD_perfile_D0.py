"""
Per-file publication MSD figures — water (D0) reference measurements.

Dissertation layout (Abb23; user, 2026-10-14), defined here in
plot_msd_dissertation() and also used by MSD_per_file_20mg.py (Abb24):
  Panel A  eMSD (log-log, first MSD_MAX_POINTS lags, x axis fitted to that lag
           range), Stokes-Einstein 4 D0 tau, power-law fit (unchanged fit and
           fit points), normalised-MSD inset lower right
  Panel B  trajectories of the tracks that enter the eMSD, in the camera aspect
           ratio (image size from the TIFF header), scale bar with its length
  Styleguide v2, width class full, PDF; both panels have the same height.
  Colours: SIZE_COLORS, eMSD markers in the base colour for buffer and in the
  dark colour for hydrogel, fit line in the dark colour, theory black dashed.
Files named in DISSERTATION_FILES are written to SAVE_PATH, all other files to
SCRATCH_PATH. The per-file summary table goes to SAVE_PATH.

Reads the per-file results already computed by MSD_FromTrackmate_D0.py
(msd_d0_results.pkl) instead of reprocessing the raw TrackMate XML files —
that script always recomputes fresh and overwrites the pickle on every run,
so run it first (or after any change to the raw data) to refresh this input.

Tracks are pre-filtered via core.io.single_file_data() -> remove_edge_artifacts():
detections within 3% of the frame border are dropped and trajectories split at
the gap, to correct spurious TrackMate linking of near-edge detections.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

from hydro_analysis.core.io import (
    get_dls_reference_maps, get_dls_sizes, get_dls_labels,
    condition_label_from_filename, parse_rec_comment_metadata, parse_chamber_day_repeat,
)
from hydro_analysis.core.analysis import DEFAULT_MSD_FIT_POINTS
from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import (
    clip_msd_points, _add_log_minor_ticks, _nice_length, MAX_TRACKS_DISPLAY,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "cache" / "msd_d0_results.pkl"

RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
SAVE_PATH = RESULTS / "SPT"
SCRATCH_PATH = RESULTS / "_scratch" / "MSD_perfile_D0"
# file shown in the dissertation -> output name (user selection, SPT_ausgewaehlt_Abbildungen.docx)
DISSERTATION_FILES = {"50 nm water_04_Tracks.xml": "emsd_buffer_50nm_water_04"}
MSD_FIT_POINTS   = DEFAULT_MSD_FIT_POINTS
MSD_MAX_POINTS   = 22    # plot only the first N lag points (eMSD, theory, fit); None = all
YLIM             = (0.01, 100.0)  # MSD (µm²), fixed for cross-file comparability
STYLE            = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
FIG_WIDTH_IN     = 6.30  # Styleguide width class "full"
COLOR_THEORY     = "black"
DASH_THEORY      = (0, (4, 3))
EMSD_MARKERSIZE  = 4.5


def plot_msd_dissertation(rd: dict, D_theo: float, label_nm: int, n_fit: int, medium: str,
                          sample_label: str, out_file: Path) -> None:
    """Panel A: eMSD with theory, fit and normalised inset; panel B: trajectories in
    the camera aspect ratio with a labelled scale bar. medium: "buffer" or "hydrogel"."""
    fit = rd.get("fit_results_MSD") or {}
    emsd, imsd = fit.get("emsd"), fit.get("imsd")
    A_fit, n_exp = fit.get("A"), fit.get("exponent")
    if emsd is None or emsd.empty:
        print(f"  [SKIP] no eMSD for {out_file.name}")
        return
    base, dark = SIZE_COLORS[float(rd["particle_size_nm"])]
    face = base if medium == "buffer" else dark
    mpp = float(rd["mpp"])
    with tifffile.TiffFile(rd["tif_path"]) as tif:
        height_px, width_px = tif.pages[0].shape[:2]
    width_um, height_um = width_px * mpp, height_px * mpp

    # geometry (inches): both panels share the plot height h; B has the camera aspect
    left, gap, right, bottom, top = 0.62, 0.30, 0.05, 0.45, 0.22
    aspect_b = width_um / height_um
    ratio_a = 1.35                                   # panel A width / height
    h = (FIG_WIDTH_IN - left - gap - right) / (ratio_a + aspect_b)
    fig_h = h + bottom + top
    W = FIG_WIDTH_IN
    with plt.style.context(STYLE):
        fig = plt.figure(figsize=(W, fig_h), layout="none")
        ax = fig.add_axes((left / W, bottom / fig_h, ratio_a * h / W, h / fig_h))
        ax_b = fig.add_axes(((left + ratio_a * h + gap) / W, bottom / fig_h, aspect_b * h / W, h / fig_h))

        lag = emsd.index.values.astype(float)
        ev = emsd.values.astype(float)
        valid = ev > 0
        t_th = np.logspace(np.log10(lag[valid].min()), np.log10(lag[valid].max()), 300)
        ax.plot(t_th, 4.0 * D_theo * t_th, color=COLOR_THEORY, linewidth=1.2, linestyle=DASH_THEORY, zorder=3)
        if A_fit is not None and n_exp is not None:
            ax.plot(t_th, float(A_fit) * t_th ** float(n_exp), color=dark, linewidth=1.5, zorder=4)
        ax.plot(lag[valid], ev[valid], linestyle="none", marker="o", markersize=EMSD_MARKERSIZE,
                markerfacecolor=face, markeredgecolor=dark, markeredgewidth=0.6, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lag[valid].min() / 1.25, lag[valid].max() * 1.25)
        ax.set_ylim(*YLIM)
        _add_log_minor_ticks(ax)
        ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        ax.set_xlabel(r"Lag time $\tau$ (s)")
        ax.set_ylabel(r"MSD (µm$^2$)")
        s = str(label_nm)
        handles = [
            Line2D([0], [0], linestyle="none", marker="o", markersize=EMSD_MARKERSIZE, markerfacecolor=face,
                   markeredgecolor=dark, markeredgewidth=0.6, label=f"eMSD ({s} nm)"),
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=DASH_THEORY,
                   label=rf"Stokes–Einstein $4D_{{0}}\tau$"),
        ]
        if A_fit is not None and n_exp is not None:
            handles.append(Line2D([0], [0], color=dark, linewidth=1.5, label=r"Power-law fit $A\tau^n$"))
        ax.legend(handles=handles, loc="upper left")
        ax.text(0.03, 0.03, sample_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)

        # normalised MSD inset, lower right (fit points only)
        axin = ax.inset_axes([0.62, 0.17, 0.35, 0.30])
        fit_mask = valid & (np.arange(len(lag)) < n_fit)
        lag_fit, ev_fit = lag[fit_mask], ev[fit_mask]
        norm_y = [1.0]
        axin.axhline(1.0, color=COLOR_THEORY, linewidth=1.0, linestyle=DASH_THEORY, zorder=3)
        if A_fit is not None and n_exp is not None and lag_fit.size >= 2:
            t_dense = np.logspace(np.log10(lag_fit[0]), np.log10(lag_fit[-1]), 150)
            y_fit_n = (float(A_fit) * t_dense ** float(n_exp)) / (4.0 * D_theo * t_dense)
            axin.plot(t_dense, y_fit_n, color=dark, linewidth=1.2, zorder=4)
            norm_y.extend(y_fit_n.tolist())
        if lag_fit.size >= 2:
            y_emsd_n = ev_fit / (4.0 * D_theo * lag_fit)
            axin.plot(lag_fit, y_emsd_n, linestyle="none", marker="o", markersize=3.5, markerfacecolor=face,
                      markeredgecolor=dark, markeredgewidth=0.5, zorder=5)
            norm_y.extend(y_emsd_n.tolist())
            axin.set_xlim(lag_fit[0] * 0.85, lag_fit[-1] * 1.15)
        axin.set_xscale("log")
        axin.set_yscale("log")
        fin_y = [v for v in norm_y if np.isfinite(v) and v > 0]
        if fin_y:
            axin.set_ylim(min(fin_y) * 0.6, max(fin_y) * 1.6)
        _add_log_minor_ticks(axin)
        axin.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        axin.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        axin.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        axin.tick_params(which="both", labelsize=7, pad=1.5)
        axin.set_xlabel(r"$\tau$ (s)", fontsize=7, labelpad=0.5)
        axin.set_ylabel(r"MSD$/4D_0\tau$", fontsize=7, labelpad=1)

        # panel B: trajectories of the tracks in the eMSD, camera field of view
        tracks_df = rd.get("tracks_df")
        if tracks_df is not None and not tracks_df.empty:
            traj_df = tracks_df[tracks_df["particle"].isin(set(imsd.columns))] if imsd is not None else tracks_df
            particles = traj_df["particle"].unique()
            if len(particles) > MAX_TRACKS_DISPLAY:
                particles = np.random.default_rng(42).choice(particles, size=MAX_TRACKS_DISPLAY, replace=False)
            for pid in particles:
                g = traj_df[traj_df["particle"] == pid].sort_values("frame")
                ax_b.plot(g["x"].to_numpy(float) * mpp, g["y"].to_numpy(float) * mpp, color="black",
                          linewidth=0.4, alpha=0.6, rasterized=True)
        ax_b.set_xlim(0, width_um)
        ax_b.set_ylim(height_um, 0)
        ax_b.set_aspect("equal")
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        ax_b.tick_params(which="both", top=False, right=False, left=False, bottom=False)
        bar_um = _nice_length(width_um / 5.0)
        x_r, y_b = width_um * 0.95, height_um * 0.93
        ax_b.plot([x_r - bar_um, x_r], [y_b, y_b], color="black", linewidth=2.0, solid_capstyle="butt")
        ax_b.text(x_r - bar_um / 2, y_b - height_um * 0.03, f"{bar_um:g} µm", ha="center", va="bottom", fontsize=8)
        for a, letter in ((ax, "A"), (ax_b, "B")):
            fig.text(a.get_position().x0 - (0.55 if a is ax else 0.05) / W, (bottom + h + 0.05) / fig_h, letter,
                     fontsize=10, fontweight="bold", ha="left", va="bottom")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file)
        plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CACHE_FILE.exists():
        print(f"Kein Cache gefunden: {CACHE_FILE}")
        print("Bitte zuerst MSD_FromTrackmate_D0.py ausführen.")
        return
    with open(CACHE_FILE, "rb") as f:
        results = pickle.load(f)
    print(f"Cache geladen: {CACHE_FILE} ({len(results)} Dateien)")

    dls_sizes     = get_dls_sizes()
    dls_labels    = get_dls_labels()
    size_override = get_dls_reference_maps()["size_override_nm"]

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    SCRATCH_PATH.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    n_processed = 0
    n_skipped   = 0

    ordered = sorted(
        results.values(),
        key=lambda r: (r.get("particle_size_nm") or 0.0, r.get("base_name", "")),
    )

    for rd in ordered:
        size_nm = rd.get("particle_size_nm")
        fit     = rd.get("fit_results_MSD")
        if size_nm is None or fit is None:
            n_skipped += 1
            continue

        mapped_size = dls_sizes.get(size_nm, size_override.get(size_nm, size_nm))
        label_nm    = dls_labels.get(size_nm, int(size_nm))
        D_theo      = calculate_theoretical_diffusion(particle_size_nm=mapped_size)

        base = rd.get("base_name", "")
        fps  = rd.get("fps")
        print(f"  [{int(size_nm)} nm  {fps:.0f} fps] {base}")

        tracks_df = rd.get("tracks_df")
        if tracks_df is not None and not tracks_df.empty:
            track_lengths          = tracks_df.groupby("particle").size()
            num_detections_total   = len(tracks_df)
            mean_track_length      = track_lengths.mean()
        else:
            num_detections_total   = 0
            mean_track_length      = np.nan

        duration_s   = rd["num_frames"] / fps if fps else np.nan
        rec_meta     = parse_rec_comment_metadata(rd.get("rec_path"))
        chamber_meta = parse_chamber_day_repeat(base)

        summary_rows.append({
            "file":                          base,
            "size_nm":                       int(size_nm),
            "label_nm":                      label_nm,
            "condition":                     condition_label_from_filename(base),
            "chamber":                       chamber_meta["chamber"],
            "day":                           chamber_meta["day"],
            "repeat":                        chamber_meta["repeat"],
            "fps":                           fps,
            "mpp_um_per_px":                 rd.get("mpp"),
            "D_MSD_um2_per_s":               fit.get("D_um2_per_s",    np.nan),
            "D_error":                       fit.get("D_error",         np.nan),
            "exponent":                      fit.get("exponent",        np.nan),
            "exponent_error":                fit.get("exponent_error",  np.nan),
            "sigma_loc_nm":                  fit.get("sigma_loc_nm",    np.nan),
            "D_theo_um2_per_s":              D_theo,
            "num_trajectories":              rd.get("num_tracks"),
            "num_detections_total":          num_detections_total,
            "mean_trajectory_length_frames": mean_track_length,
            "num_frames":                    rd.get("num_frames"),
            "duration_s":                    duration_s,
            "laser_power_mW":                rec_meta["laser_power_mW"],
            "depth_um":                      rec_meta["depth_um"],
            "recorded_at":                   rec_meta["recorded_at"],
            "xml_path":                      rd.get("xml_path"),
            "tif_path":                      rd.get("tif_path"),
            "rec_path":                      rd.get("rec_path"),
        })

        rd_plot = clip_msd_points(rd, MSD_MAX_POINTS)
        suffix  = f"_first{MSD_MAX_POINTS}" if MSD_MAX_POINTS is not None else ""
        if base in DISSERTATION_FILES:
            out_file = SAVE_PATH / f"{DISSERTATION_FILES[base]}.pdf"
        else:
            out_file = SCRATCH_PATH / f"{base}_{int(size_nm)}nm{suffix}.pdf"
        plot_msd_dissertation(rd_plot, D_theo, label_nm, MSD_FIT_POINTS, "buffer", "Buffer ($D_0$)", out_file)
        n_processed += 1

    # ── per-file summary CSV ──────────────────────────────────────────────────
    # sep=";" / decimal="," so Excel (German locale) opens this directly with
    # columns already split and numbers recognized, instead of dumping everything
    # into one column. utf-8-sig BOM so Excel auto-detects the encoding.
    if summary_rows:
        csv_path = SAVE_PATH / "per_file_summary_d0.csv"
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(csv_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
        print(f"\nSummary saved: {csv_path}")

    print(f"\nDone — {n_processed} files processed, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
