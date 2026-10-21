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

Dissertation version (user, 2026-10-14; Styleguide v2, width class narrow, PDF,
SAVE_PATH): x positions at the DLS diameters in full precision (get_dls_sizes()),
hydrogel markers in the dark SIZE_COLORS colour of each size, DLS reference as a
square #da00bd with x and y error bars, "Buffer" instead of "Water".

Run MSD_FromTrackmate_20mg.py first (or after any change to the raw data)
to refresh msd_20mg_result.pkl.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, FuncFormatter, LogLocator

from hydro_analysis.core.io import get_dls_reference_maps, get_dls_sizes, get_dls_labels
from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _DASH_THEORY, _add_log_minor_ticks
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "cache" / "msd_20mg_result.pkl"
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\SPT"
)
STYLE = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
FIG_SIZE = (4.72, 4.72 / 1.42)       # Styleguide width class "narrow"

X_MIN, X_MAX = 15.0, 1500.0

# Hydrogel: dark colour of each particle size (SIZE_COLORS), Styleguide section 11
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

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIG_SIZE)

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
        ax.errorbar(dls_x, dls_y, xerr=dls_xerr, yerr=dls_yerr, fmt="s", markersize=5,
                    markerfacecolor=COLOR_DLS, markeredgecolor=COLOR_DLS_DARK, markeredgewidth=0.6,
                    ecolor=COLOR_DLS_DARK, elinewidth=0.8, capsize=2.0, capthick=0.8,
                    linestyle="None", zorder=8)

        # ── Weighted average per size (conditions pooled) -- one point each ─
        sizes = sorted(averages.keys())
        xs    = [_map_size(s) for s in sizes]
        ys    = [(averages[s].get("fit_results_MSD") or {}).get("D_um2_per_s", np.nan) for s in sizes]
        yerrs = [(averages[s].get("fit_results_MSD") or {}).get("D_error", np.nan) for s in sizes]
        yerrs = [e if np.isfinite(e) else 0.0 for e in yerrs]
        for x, y, yerr, s in zip(xs, ys, yerrs, sizes):
            dark = SIZE_COLORS[s][1]
            ax.errorbar([x], [y], yerr=[yerr], fmt="o", markersize=5,
                        markerfacecolor=dark, markeredgecolor=dark, markeredgewidth=0.6,
                        ecolor=dark, elinewidth=0.8, capsize=2.0, capthick=0.8, linestyle="None", zorder=6)

        # ── Axes: DLS-mapped tick positions, DLS labels (not nominal names) ──
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(X_MIN, X_MAX)
        _add_log_minor_ticks(ax)
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))

        tick_nominal = sorted(dls_labels.keys())
        tick_pos     = [_map_size(s) for s in tick_nominal]
        tick_text    = [str(dls_labels[s]) for s in tick_nominal]
        ax.xaxis.set_major_locator(FixedLocator(tick_pos))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_text))

        ax.set_xlabel("Particle size (nm)")
        ax.set_ylabel(r"Diffusion coefficient $D$ (µm$^2$ s$^{-1}$)")

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label=r"Stokes–Einstein $D_0$"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_DLS,
                   markeredgecolor=COLOR_DLS_DARK, markersize=5, markeredgewidth=0.6,
                   label=r"DLS $D_0$ (buffer)", linestyle="None"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="0.3",
                   markeredgecolor="0.3", markersize=5, markeredgewidth=0.6,
                   label=r"Hydrogel 20 mg mL$^{-1}$ (weighted average)", linestyle="None"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            pdf_path = SAVE_PATH / "diffusion_vs_size_20mg_weighted_avg.pdf"
            fig.savefig(pdf_path)
            print(f"Plot gespeichert: {pdf_path}")


if __name__ == "__main__":
    main()
