"""
Diffusion coefficient D vs. particle size -- D0 (water) SPT measurements
with DLS mean reference.

Pure consumer: reads msd_d0_results.pkl (produced by MSD_FromTrackmate_D0.py)
for the per-file SPT D0 points, and hydro_analysis/Litesizer/cache/
dls_reference.pkl (via core.io.get_dls_diffusion_means()) for the DLS mean D
reference per particle size. Never touches raw TrackMate XML/TIFF/.rec files.
If msd_d0_results.pkl is missing or stale, run MSD_FromTrackmate_D0.py first.
If the DLS cache is missing, rerun the Litesizer pipeline
(litesizer_measurements_mean.py) first.

Dissertation-figure sibling of ../MSD_Diffusion_vs_Size_D0.py: same per-file
D0 points (fit_results_MSD.D_um2_per_s, alpha=0.5 so overlapping files at
one size read as a denser patch) and the same Stokes-Einstein theory line,
but the DLS reference point/error now comes from the new canonical
get_dls_diffusion_means() (D_mean_um2s / sigma_D_um2s) instead of the legacy
hardcoded get_dls_reference_maps()['dls_D_um2_per_s'] -- this is the single
canonical DLS-mean D source for all Dissertation_Figures scripts. Particle
size on the x-axis is still the real DLS z-average diameter
(get_dls_sizes()/get_dls_labels()), never the nominal folder name; the DLS
star's x-error bar still uses get_dls_reference_maps()['size_err_nm'] (a
diameter uncertainty, not superseded by the new function, which does not
carry a size-error field).

Shows the figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\D0_Diffusion_vs_Size\\
subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

from hydro_analysis.core.io import (
    get_dls_reference_maps, get_dls_sizes, get_dls_labels, get_dls_diffusion_means,
)
from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY, _add_log_minor_ticks

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "D0_Diffusion_vs_Size"

X_MIN, X_MAX = 15.0, 1500.0

COLOR_MEASURED      = "#3B6E8C"   # single-series D0 (Style_guide.txt §12)
COLOR_MEASURED_DARK = "#2A4F66"
COLOR_DLS           = "#da00bd"   # Style-guide Markierung A base
COLOR_DLS_DARK      = "#9b5191"
COLOR_THEORY        = "black"

POINT_ALPHA = 0.5


def load_results() -> dict:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst MSD_FromTrackmate_D0.py ausführen."
        )
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def main() -> None:
    results = load_results()
    print(f"Cache geladen: {CACHE_FILE} ({len(results)} Dateien)")

    dls_sizes     = get_dls_sizes()               # {nominal_nm: real DLS z-average diameter}
    dls_labels    = get_dls_labels()               # {nominal_nm: DLS label for axis display}
    dls_maps      = get_dls_reference_maps()
    size_override = dls_maps["size_override_nm"]   # fallback if a size is missing from dls_sizes
    size_err      = dls_maps["size_err_nm"]
    dls_means     = get_dls_diffusion_means()       # {nominal_nm: {D_mean_um2s, sigma_D_um2s, ...}}

    def _map_size(size: float) -> float:
        return float(dls_sizes.get(size, size_override.get(size, size)))

    def _sz_err(size: float) -> float:
        return float(size_err.get(size, 0.0))

    # ── Collect one (size, D, D_err) row per file ───────────────────────────
    rows: list[tuple[float, float, float]] = []
    n_skipped = 0
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        fit = r.get("fit_results_MSD")
        D = fit.get("D_um2_per_s") if fit else None
        if size_nm is None or D is None:
            n_skipped += 1
            continue
        rows.append((float(size_nm), float(D), float(fit.get("D_error", np.nan))))
    print(f"{len(rows)} Dateien mit gültigem D-Wert, {n_skipped} übersprungen.")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

        # ── Stokes-Einstein theory line for D0 ──────────────────────────────
        sizes_range = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 200)
        D_theory_range = [calculate_theoretical_diffusion(s) for s in sizes_range]
        ax.plot(sizes_range, D_theory_range, color=COLOR_THEORY, linewidth=1.2,
                linestyle=_DASH_THEORY, zorder=3)

        # ── DLS-measured D0 (water) mean reference points ───────────────────
        dls_ref_sizes = sorted(dls_means.keys())
        dls_x    = [_map_size(s) for s in dls_ref_sizes]
        dls_y    = [dls_means[s]["D_mean_um2s"] for s in dls_ref_sizes]
        dls_xerr = [_sz_err(s) for s in dls_ref_sizes]
        dls_yerr = [dls_means[s]["sigma_D_um2s"] for s in dls_ref_sizes]
        ax.errorbar(dls_x, dls_y, xerr=dls_xerr, yerr=dls_yerr, fmt="*", markersize=10,
                    markerfacecolor=COLOR_DLS, markeredgecolor=COLOR_DLS_DARK, markeredgewidth=0.8,
                    ecolor=COLOR_DLS_DARK, elinewidth=0.8, capsize=2.0, capthick=0.8,
                    linestyle="None", zorder=8)

        # ── Per-file D0 values -- exact x, no offset, no y error bars (file-wise plot) ──
        # Alpha lets overlapping files at the same size show up as denser/darker.
        if rows:
            xs = [_map_size(s) for s, D, Derr in rows]
            ys = [D for s, D, Derr in rows]
            ax.scatter(xs, ys, s=36, alpha=POINT_ALPHA, marker="o",
                       facecolor=COLOR_MEASURED, edgecolor=COLOR_MEASURED_DARK,
                       linewidth=0.6, zorder=5)

        # ── Axes: DLS-mapped tick positions, DLS labels (not nominal names) ──
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(X_MIN, X_MAX)
        _add_log_minor_ticks(ax)

        tick_nominal = sorted(dls_labels.keys())
        tick_pos     = [_map_size(s) for s in tick_nominal]
        tick_text    = [str(dls_labels[s]) for s in tick_nominal]
        ax.xaxis.set_major_locator(FixedLocator(tick_pos))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_text))

        ax.set_xlabel("Particle size (nm)")
        ax.set_ylabel(r"Diffusion coefficient $D$ (µm²/s)")

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label="Stokes–Einstein theory (D0)"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor=COLOR_DLS,
                   markeredgecolor=COLOR_DLS_DARK, markersize=10, markeredgewidth=0.8,
                   label="DLS D0 (water, mean)", linestyle="None"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MEASURED,
                   markeredgecolor=COLOR_MEASURED_DARK, markersize=7, markeredgewidth=0.6,
                   label="SPT D0 (water, per file)", linestyle="None"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / "d0_diffusion_vs_size.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
