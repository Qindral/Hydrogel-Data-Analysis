"""
Anomalous diffusion exponent n vs. particle size — 20 mg/mL C16 hydrogel,
trajectory-weighted average per size (Surface loading and Injection pooled).

Companion plot to MSD_Diffusion_vs_Size_20mg_WeightedAvg.py: same cache and
data-point positions, but a linear 0-100 nm x-axis (as Abb09, user, 2026-10-14) with uniform tick spacing
(rather than that script's log axis with DLS-mapped tick positions), and
plots the eMSD anomalous exponent n (MSD ~ tau^n) instead of D. One point
per particle size, weighted by each
file's trajectory count -- read directly from msd_20mg_result.pkl, produced
by MSD_FromTrackmate_20mg.py's weighted_average_per_size(), so there is only
one weighted-average calculation in the whole pipeline.

Overlaid for reference: n = 1 (free diffusion).

For a fitted obstruction model (quadratic n(r) fit with MC uncertainty band),
see MSD_AnomalousExponent_SaxtonFit.py instead -- this script only shows the
raw weighted-average data points.

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
from matplotlib.ticker import MultipleLocator

from hydro_analysis.core.io import get_dls_reference_maps, get_dls_sizes
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _DASH_THEORY
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "cache" / "msd_20mg_result.pkl"
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\SPT"
)
STYLE = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
FIG_SIZE = (4.72, 4.72 / 1.42)       # Styleguide width class "narrow"

X_MIN, X_MAX = 0.0, 100.0   # as Abb09, decided by user, 2026-10-14
Y_MIN, Y_MAX = 0.0, 1.1

# Hydrogel: dark colour of each particle size (SIZE_COLORS), Styleguide section 11
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
              f"n = {fit.get('exponent', float('nan')):.3g} ± {fit.get('exponent_error', float('nan')):.2g}  "
              f"(N_files={entry.get('n_files')}, N_particles={entry.get('n_particles_pooled')})")

    dls_sizes     = get_dls_sizes()               # {nominal_nm: real DLS z-average diameter}
    dls_maps      = get_dls_reference_maps()
    size_override = dls_maps["size_override_nm"]   # fallback if a size is missing from dls_sizes

    def _map_size(size: float) -> float:
        return float(dls_sizes.get(size, size_override.get(size, size)))

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        # ── Reference: free diffusion (n = 1) ───────────────────────────────
        ax.axhline(1.0, color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY, zorder=3)

        # ── Weighted average per size (conditions pooled) -- one point each ─
        sizes = sorted(averages.keys())
        xs    = [_map_size(s) for s in sizes]
        ys    = [(averages[s].get("fit_results_MSD") or {}).get("exponent", np.nan) for s in sizes]
        yerrs = [(averages[s].get("fit_results_MSD") or {}).get("exponent_error", np.nan) for s in sizes]
        yerrs = [e if np.isfinite(e) else 0.0 for e in yerrs]
        for x, y, yerr, s in zip(xs, ys, yerrs, sizes):
            dark = SIZE_COLORS[s][1]
            ax.errorbar([x], [y], yerr=[yerr], fmt="o", markersize=5,
                        markerfacecolor=dark, markeredgecolor=dark, markeredgewidth=0.6,
                        ecolor=dark, elinewidth=0.8, capsize=2.0, capthick=0.8, linestyle="None", zorder=6)

        # ── Axes: linear, uniform tick spacing (data still sits at the ──────
        # DLS-mapped size; only the tick grid itself is evenly spaced) ──────
        ax.set_xscale("linear")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        ax.set_xlabel("Particle size (nm)")
        ax.set_ylabel(r"Anomalous exponent $n$")

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label="Free diffusion (n = 1)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="0.3",
                   markeredgecolor="0.3", markersize=5, markeredgewidth=0.6,
                   label=r"Hydrogel 20 mg mL$^{-1}$ (weighted average)", linestyle="None"),
        ]
        ax.legend(handles=legend_elements, loc="lower left", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            pdf_path = SAVE_PATH / "anomalous_exponent_vs_size_20mg_weighted_avg.pdf"
            fig.savefig(pdf_path)
            print(f"Plot gespeichert: {pdf_path}")


if __name__ == "__main__":
    main()
