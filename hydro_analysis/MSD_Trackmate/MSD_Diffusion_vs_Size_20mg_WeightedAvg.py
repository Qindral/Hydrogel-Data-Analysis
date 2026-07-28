"""
Diffusion coefficient D vs. particle size — 20 mg/mL C16 hydrogel,
trajectory-weighted average per size (Surface loading and Injection pooled).

Same chart as MSD_Diffusion_vs_Size_20mg.py, but instead of one point per
file (colored by condition) it shows one point per particle size: Surface
loading and Injection files are taken together and averaged, weighted by
each file's trajectory count. This is exactly msd_20mg_result.pkl, produced
by MSD_FromTrackmate_20mg.py's weighted_average_per_size() -- read directly
here rather than recomputed, so there is only one weighted-average
calculation in the whole pipeline.

Overlaid for reference (same as MSD_Diffusion_vs_Size_20mg.py):
  - the DLS-measured D0 (water) points at each particle size
  - the continuous Stokes-Einstein theory line for D0

Run MSD_FromTrackmate_20mg.py first (or after any change to the raw data)
to refresh msd_20mg_result.pkl.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

from hydro_analysis.core.io import get_dls_reference_maps, get_dls_sizes, get_dls_labels
from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY, _add_log_minor_ticks

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "cache" / "msd_20mg_result.pkl"
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)

X_MIN, X_MAX = 15.0, 1500.0

# Same style as MSD_Diffusion_vs_Size_20mg.py (Style_guide.txt §12). Single
# series now (Surface loading + Injection pooled), so one muted color.
COLOR_MEASURED      = "#3B8C8C"   # desaturated green-blue (teal)
COLOR_MEASURED_DARK = "#2A6666"
COLOR_DLS           = "#da00bd"   # Style-guide Markierung A base
COLOR_DLS_DARK      = "#9b5191"
COLOR_THEORY        = "black"


def load_averages() -> dict:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst MSD_FromTrackmate_20mg.py ausführen."
        )
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def main() -> None:
    averages = load_averages()
    print(f"Cache geladen: {CACHE_FILE} ({len(averages)} Größen)")
    for size_nm, entry in sorted(averages.items()):
        fit = entry.get("fit_results_MSD") or {}
        print(f"  {int(size_nm):>5} nm  "
              f"D = {fit.get('D_um2_per_s', float('nan')):.4g} ± {fit.get('D_error', float('nan')):.2g} µm²/s  "
              f"(N_files={entry.get('n_files')}, N_particles={entry.get('n_particles_pooled')})")

    dls_sizes     = get_dls_sizes()               # {nominal_nm: real DLS z-average diameter}
    dls_labels    = get_dls_labels()               # {nominal_nm: DLS label for axis display}
    dls_maps      = get_dls_reference_maps()
    size_override = dls_maps["size_override_nm"]   # fallback if a size is missing from dls_sizes
    size_err      = dls_maps["size_err_nm"]
    dls_D         = dls_maps["dls_D_um2_per_s"]
    dls_D_err     = dls_maps["dls_D_err_um2_per_s"]

    def _map_size(size: float) -> float:
        return float(dls_sizes.get(size, size_override.get(size, size)))

    def _sz_err(size: float) -> float:
        return float(size_err.get(size, 0.0))

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

        # ── Stokes-Einstein theory line for D0 ──────────────────────────────
        sizes_range = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 200)
        D_theory_range = [calculate_theoretical_diffusion(s) for s in sizes_range]
        ax.plot(sizes_range, D_theory_range, color=COLOR_THEORY, linewidth=1.2,
                linestyle=_DASH_THEORY, zorder=3)

        # ── DLS-measured D0 (water) reference points ────────────────────────
        dls_ref_sizes = sorted(dls_D.keys())
        dls_x    = [_map_size(s) for s in dls_ref_sizes]
        dls_y    = [dls_D[s] for s in dls_ref_sizes]
        dls_xerr = [_sz_err(s) for s in dls_ref_sizes]
        dls_yerr = [dls_D_err.get(s, 0.0) for s in dls_ref_sizes]
        ax.errorbar(dls_x, dls_y, xerr=dls_xerr, yerr=dls_yerr, fmt="*", markersize=10,
                    markerfacecolor=COLOR_DLS, markeredgecolor=COLOR_DLS_DARK, markeredgewidth=0.8,
                    ecolor=COLOR_DLS_DARK, elinewidth=0.8, capsize=2.0, capthick=0.8,
                    linestyle="None", zorder=8)

        # ── Weighted average per size (conditions pooled) -- one point each ─
        sizes = sorted(averages.keys())
        xs    = [_map_size(s) for s in sizes]
        ys    = [(averages[s].get("fit_results_MSD") or {}).get("D_um2_per_s", np.nan) for s in sizes]
        yerrs = [(averages[s].get("fit_results_MSD") or {}).get("D_error", np.nan) for s in sizes]
        yerrs = [e if np.isfinite(e) else 0.0 for e in yerrs]
        ax.errorbar(xs, ys, yerr=yerrs, fmt="o", markersize=9,
                    markerfacecolor=COLOR_MEASURED, markeredgecolor=COLOR_MEASURED_DARK,
                    markeredgewidth=0.8, ecolor=COLOR_MEASURED_DARK, elinewidth=1.0,
                    capsize=3.0, capthick=1.0, linestyle="None", zorder=6)

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
                   label="DLS D0 (water)", linestyle="None"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MEASURED,
                   markeredgecolor=COLOR_MEASURED_DARK, markersize=8, markeredgewidth=0.8,
                   label="20 mg/mL (weighted avg., Surface loading + Injection)", linestyle="None"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / "diffusion_vs_size_20mg_weighted_avg.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
