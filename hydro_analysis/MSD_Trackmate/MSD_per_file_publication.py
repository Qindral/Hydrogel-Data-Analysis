"""
Per-file publication MSD figures for single-particle tracking.

Loops through all configured XML folders, processes each TrackMate XML
file individually (no aggregation), and saves a publication figure +
per-file CSV table.

Figure layout
─────────────
  Main axes      iMSD (grey) · eMSD (blue, markers) · Stokes-Einstein D₀
                 (black dashed) · power-law fit (red)
  Legend         upper left
  Upper-right    normalised-MSD inset  MSD / (4 D₀ τ)
  Lower-right    raw-trajectory inset  (plain, scale bar only, no text)
  Annotation     d = <label> nm  — lower-right corner of main axes

Tracks are pre-filtered via core.io.single_file_data() -> remove_edge_artifacts():
detections within 3% of the frame border are dropped and trajectories split at
the gap, to correct spurious TrackMate linking of near-edge detections.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from hydro_analysis.core.io import single_file_data, get_dls_reference_maps, get_dls_sizes, get_dls_labels
from hydro_analysis.core.analysis import (
    perform_msd_analysis,
    DEFAULT_MSD_FIT_POINTS,
)
from hydro_analysis.core.physics import calculate_theoretical_diffusion

# ── Configuration ──────────────────────────────────────────────────────────────
XML_FOLDERS: dict[float, list[Path]] = {
    20.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\20 nm\Tracks"),
        Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.16\Tracks_20"),
    ],
    50.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_50"),
    ],
    100.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\100 nm\Tracks"),
    ],
    200.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\200 nm\Tracks"),
    ],
    500.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\500 nm\Tracks"),
    ],
    1000.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_1000"),
    ],
}

SAVE_PATH = Path(
    rf"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung"
    rf"\PerFile_{pd.Timestamp.now().strftime('%Y%m%d')}"
)
MSD_FIT_POINTS    = DEFAULT_MSD_FIT_POINTS
FPS_SMALL_TARGET  = 60.0
FPS_LARGE_TARGET  = 20.0
FPS_TOLERANCE     = 40.0
MAX_TRACKS_DISPLAY = 300  # max particle tracks drawn in trajectory inset

# ── Style ───────────────────────────────────────────────────────────────────────
_COL_IMSD    = "black"
_COL_EMSD    = "#1a1aee"
_COL_THEORY  = "black"
_COL_FIT     = "#cc2200"
_COL_TRK     = "#000000"

_ALPHA_IMSD  = 0.1
_LW_IMSD     = 0.65
_LW_EMSD     = 0.8
_MS_EMSD     = 3
_LW_THEORY   = 1.2
_LW_FIT      = 1.0
_DASH_THEORY = (0, (4, 3))

_RC = {
    "text.usetex":       False,
    "mathtext.fontset":  "dejavusans",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Open Sans", "Arial", "DejaVu Sans"],
    "axes.linewidth":    0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.major.size":  5.0,
    "ytick.major.size":  5.0,
    "xtick.minor.size":  3.0,
    "ytick.minor.size":  3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "legend.fontsize":   8,
}


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _add_log_minor_ticks(ax: plt.Axes) -> None:
    loc = mticker.LogLocator(subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=100)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(loc)
        axis.set_minor_formatter(mticker.NullFormatter())


def _nice_length(target: float) -> float:
    """Round target to the nearest 1 / 2 / 5 × 10^n."""
    if target <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(target))
    norm = target / mag
    if norm < 1.5:
        return 1.0 * mag
    if norm < 3.5:
        return 2.0 * mag
    if norm < 7.5:
        return 5.0 * mag
    return 10.0 * mag


def _draw_scalebar(ax: plt.Axes, length_um: float) -> None:
    """Draw a plain scale-bar line in the lower-right corner (no text)."""
    ax.autoscale_view()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    x_r = xlim[1] - 0.06 * x_range
    x_l = x_r - length_um
    y_p = ylim[0] + 0.06 * y_range
    ax.plot([x_l, x_r], [y_p, y_p],
            color="black", linewidth=2.0, solid_capstyle="butt",
            transform=ax.transData)


def _inset_style(ax: plt.Axes) -> None:
    ax.tick_params(which="both", direction="in",
                   top=True, right=True, labelsize=7, width=0.6)
    ax.tick_params(which="major", length=3.5)
    ax.tick_params(which="minor", length=2.0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)




# ── Plot ────────────────────────────────────────────────────────────────────────

def plot_msd_single_file(
    rd: dict,
    D_theo: float,
    label_nm: int,
    n_fit: int = DEFAULT_MSD_FIT_POINTS,
    save_path: Path | str | None = None,
    filename: str = "msd",
    xlim: tuple = (1e-2, 4.0),
    sample_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Publication figure for one TrackMate XML file.

    Parameters
    ----------
    rd           result_dict from single_file_data + perform_msd_analysis
    D_theo       Stokes-Einstein diffusion coefficient (µm²/s)
    label_nm     DLS-corrected particle diameter label (nm) for the annotation
    n_fit        number of lag-time points used for the power-law fit
    save_path    directory; saves <filename>.pdf and <filename>.png (300 dpi)
    filename     base name without extension
    xlim         (x_min, x_max) lag-time axis limits in seconds
    sample_label short sample description shown inside the figure
    """
    fit = rd.get("fit_results_MSD") or {}
    imsd: pd.DataFrame | None = fit.get("imsd")
    emsd: pd.Series | None    = fit.get("emsd")
    tracks_df: pd.DataFrame | None = rd.get("tracks_df")
    mpp  = float(rd.get("mpp", 1.0))
    A_fit  = fit.get("A")
    n_exp  = fit.get("exponent")

    if emsd is None or emsd.empty:
        print(f"  [SKIP] no eMSD for {filename}")
        return None, None

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))

        lag   = emsd.index.values.astype(float)
        ev    = emsd.values.astype(float)
        valid = ev > 0

        # ── theory line ───────────────────────────────────────────────────
        t_th = np.logspace(np.log10(lag[valid].min()),
                           np.log10(lag[valid].max()), 300)
        ax.plot(t_th, 4.0 * D_theo * t_th,
                color=_COL_THEORY, linewidth=_LW_THEORY,
                linestyle=_DASH_THEORY, zorder=3)

        # ── fit line (full range) ─────────────────────────────────────────
        if A_fit is not None and n_exp is not None:
            ax.plot(t_th, float(A_fit) * t_th ** float(n_exp),
                    color=_COL_FIT, linewidth=_LW_FIT, zorder=4)

        # ── individual MSDs ───────────────────────────────────────────────
        if imsd is not None and not imsd.empty:
            for col in imsd.columns:
                s = imsd[col].dropna()
                s = s[s > 0]
                if not s.empty:
                    ax.plot(s.index.values, s.values,
                            color=_COL_IMSD, linewidth=_LW_IMSD,
                            alpha=_ALPHA_IMSD, zorder=1, rasterized=True)

        # ── ensemble MSD ──────────────────────────────────────────────────
        ax.plot(lag[valid], ev[valid],
                color=_COL_EMSD, linewidth=_LW_EMSD, zorder=5,
                marker="o", markersize=_MS_EMSD, markeredgewidth=0)

        # ── axis formatting (fixed ranges for cross-file comparability) ───
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(1e-2, 1e3)
        _add_log_minor_ticks(ax)
        ax.set_xlabel(r"Lag time $\tau$ (s)")
        ax.set_ylabel(r"MSD ($\mu\mathrm{m}^2$)")

        # ── legend upper left ─────────────────────────────────────────────
        s = str(label_nm) if label_nm is not None else ""
        imsd_label = f"iMSD ({s} nm)" if s else "individual MSD"
        emsd_label = f"eMSD ({s} nm)" if s else "ensemble MSD"
        se_label   = (rf"Stokes–Einstein $4D_{{0}}^{{{s}\,\mathrm{{nm}}}}\tau$"
                      if s else r"Stokes–Einstein ($4D_0\tau$)")
        handles = [
            Line2D([0], [0], color=_COL_IMSD, linewidth=0.9, alpha=0.5,
                   label=imsd_label),
            Line2D([0], [0], color=_COL_EMSD, linewidth=_LW_EMSD,
                   marker="o", markersize=_MS_EMSD, markeredgewidth=0,
                   label=emsd_label),
            Line2D([0], [0], color=_COL_THEORY, linewidth=_LW_THEORY,
                   linestyle=_DASH_THEORY, label=se_label),
        ]
        if A_fit is not None and n_exp is not None:
            handles.append(
                Line2D([0], [0], color=_COL_FIT, linewidth=_LW_FIT,
                       label=r"power-law fit ($A\tau^n$)"),
            )
        ax.legend(handles=handles, loc="upper left", frameon=False)

        if sample_label:
            ax.text(0.03, 0.03, sample_label, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8, fontstyle="italic")

        # ── upper-right inset: normalised MSD ────────────────────────────
        axin_fit = ax.inset_axes([0.60, 0.54, 0.36, 0.36])

        fit_mask = valid & (np.arange(len(lag)) < n_fit)
        lag_fit  = lag[fit_mask]
        ev_fit   = ev[fit_mask]

        norm_y_all = [1.0]
        axin_fit.axhline(1.0, color=_COL_THEORY, linewidth=1.0,
                         linestyle=_DASH_THEORY, zorder=3)

        if A_fit is not None and n_exp is not None and lag_fit.size >= 2:
            t_dense = np.logspace(np.log10(lag_fit[0]), np.log10(lag_fit[-1]), 150)
            y_fit_n = (float(A_fit) * t_dense ** float(n_exp)) / (4.0 * D_theo * t_dense)
            axin_fit.plot(t_dense, y_fit_n,
                          color=_COL_FIT, linewidth=_LW_FIT, zorder=4)
            norm_y_all.extend(y_fit_n.tolist())

        if lag_fit.size >= 2:
            y_emsd_n = ev_fit / (4.0 * D_theo * lag_fit)
            axin_fit.plot(lag_fit, y_emsd_n,
                          color=_COL_EMSD, linewidth=_LW_EMSD, zorder=5,
                          marker="o", markersize=_MS_EMSD, markeredgewidth=0)
            norm_y_all.extend(y_emsd_n.tolist())

        axin_fit.set_xscale("log")
        axin_fit.set_yscale("log")
        _add_log_minor_ticks(axin_fit)
        _inset_style(axin_fit)

        if lag_fit.size >= 2:
            axin_fit.set_xlim(lag_fit[0] * 0.85, lag_fit[-1] * 1.15)
        fin_y = [v for v in norm_y_all if np.isfinite(v) and v > 0]
        if fin_y:
            axin_fit.set_ylim(min(fin_y) * 0.5, max(fin_y) * 2.0)
        _inset_ylabel = (rf"MSD $/ 4D_0^{{{s}\,\mathrm{{nm}}}}\tau$"
                         if s else r"MSD $/ 4D_0\tau$")
        axin_fit.text(0.97, 0.04, r"$\tau$ (s)", transform=axin_fit.transAxes,
                      ha="right", va="bottom", fontsize=7)
        axin_fit.text(0.04, 0.96, _inset_ylabel, transform=axin_fit.transAxes,
                      ha="left", va="top", fontsize=7)

        # ── lower-right inset: raw trajectories ──────────────────────────
        axin_trk = ax.inset_axes([0.60, 0.10, 0.36, 0.36])

        if tracks_df is not None and not tracks_df.empty:
            # Only show particles that passed into the iMSD/eMSD computation
            if imsd is not None and not imsd.empty:
                valid_pids = set(imsd.columns)
                traj_df = tracks_df[tracks_df["particle"].isin(valid_pids)]
            else:
                traj_df = tracks_df

            particles = traj_df["particle"].unique()
            if len(particles) > MAX_TRACKS_DISPLAY:
                rng = np.random.default_rng(42)
                particles = rng.choice(particles, size=MAX_TRACKS_DISPLAY, replace=False)

            for pid in particles:
                g = traj_df[traj_df["particle"] == pid].sort_values("frame")
                xu = g["x"].to_numpy(dtype=float) * mpp
                yu = g["y"].to_numpy(dtype=float) * mpp
                axin_trk.plot(xu, yu, color=_COL_TRK,
                              linewidth=0.4, alpha=0.55, rasterized=True)

            # Equal aspect so µm scale is not distorted
            axin_trk.set_aspect("equal", adjustable="datalim")

            # Strip all axis decorations
            axin_trk.set_xticks([])
            axin_trk.set_yticks([])
            for sp in axin_trk.spines.values():
                sp.set_linewidth(0.5)

            # Scale bar: ~1/5 of the x range, rounded to a nice number
            all_x = traj_df["x"].to_numpy(dtype=float) * mpp
            x_range = float(all_x.max() - all_x.min())
            bar_um = _nice_length(x_range / 5.0)
            _draw_scalebar(axin_trk, bar_um)
        else:
            axin_trk.set_visible(False)

        # ── save ──────────────────────────────────────────────────────────
        fig.tight_layout(pad=0.4)

        if save_path is not None:
            out = Path(save_path)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{filename}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        return fig, ax


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    dls_sizes  = get_dls_sizes()
    dls_labels = get_dls_labels()
    size_override = get_dls_reference_maps()["size_override_nm"]

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    n_processed = 0
    n_skipped   = 0

    for size_nm, folders in XML_FOLDERS.items():
        target_fps  = FPS_SMALL_TARGET if size_nm < 200 else FPS_LARGE_TARGET
        mapped_size = dls_sizes.get(size_nm, size_override.get(size_nm, size_nm))
        label_nm    = dls_labels.get(size_nm, int(size_nm))
        D_theo      = calculate_theoretical_diffusion(particle_size_nm=mapped_size)

        for folder in folders:
            if not folder.exists():
                print(f"WARNING: folder not found: {folder}")
                continue

            xml_files = sorted(folder.glob("*.xml"))
            if not xml_files:
                print(f"  No XML files in {folder}")
                continue

            for xml_path in xml_files:
                rd = single_file_data(xml_path)
                if rd is None:
                    n_skipped += 1
                    continue

                fps = rd.get("fps")
                if fps is None or abs(float(fps) - target_fps) > FPS_TOLERANCE:
                    print(f"  [SKIP fps={fps:.0f}] {xml_path.name}")
                    n_skipped += 1
                    continue

                rd["particle_size_nm"] = size_nm
                perform_msd_analysis(rd, fit_points=MSD_FIT_POINTS)

                fit  = rd.get("fit_results_MSD") or {}
                base = rd.get("base_name", xml_path.stem)
                print(f"  [{int(size_nm)} nm  {fps:.0f} fps] {base}")

                summary_rows.append({
                    "file":             base,
                    "size_nm":          int(size_nm),
                    "label_nm":         label_nm,
                    "fps":              fps,
                    "D_MSD_um2_per_s":  fit.get("D_um2_per_s", np.nan),
                    "D_error":          fit.get("D_error",      np.nan),
                    "exponent":         fit.get("exponent",     np.nan),
                    "exponent_error":   fit.get("exponent_error", np.nan),
                    "r_squared":        fit.get("r_squared",    np.nan),
                    "sigma_loc_nm":     fit.get("sigma_loc_nm", np.nan),
                    "D_theo_um2_per_s": D_theo,
                })

                xlim = (0.01, 6.0) if target_fps == FPS_SMALL_TARGET else (0.04, 3.0)
                plot_msd_single_file(
                    rd=rd,
                    D_theo=D_theo,
                    label_nm=label_nm,
                    n_fit=MSD_FIT_POINTS,
                    save_path=SAVE_PATH,
                    filename=f"{base}_{int(size_nm)}nm",
                    xlim=xlim,
                )
                n_processed += 1

    # ── per-file summary CSV ──────────────────────────────────────────────────
    # sep=";" / decimal="," so Excel (German locale) opens this directly with
    # columns already split and numbers recognized, instead of dumping everything
    # into one column. utf-8-sig BOM so Excel auto-detects the encoding.
    if summary_rows and SAVE_PATH is not None:
        csv_path = SAVE_PATH / "per_file_summary.csv"
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(csv_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
        print(f"\nSummary saved: {csv_path}")

    if summary_rows:
        print("\nPower-law fit quality (R²):")
        header = f"{'file':<40} {'size_nm':>7} {'exponent':>9} {'R²':>8}"
        print(header)
        print("-" * len(header))
        for row in summary_rows:
            r2 = row["r_squared"]
            r2_str = f"{r2:.4f}" if np.isfinite(r2) else "n/a"
            exp_str = f"{row['exponent']:.3f}" if np.isfinite(row["exponent"]) else "n/a"
            print(f"{row['file']:<40} {row['size_nm']:>7} {exp_str:>9} {r2_str:>8}")

    print(f"\nDone — {n_processed} files processed, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
