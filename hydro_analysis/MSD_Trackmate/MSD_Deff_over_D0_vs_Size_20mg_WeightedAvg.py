"""
SPT D_eff / DLS D0 vs. particle size — 20 mg/mL C16 hydrogel,
trajectory-weighted average per size (Surface loading and Injection pooled).

Same style as MSD_Diffusion_vs_Size_20mg_WeightedAvg.py (DLS-mapped tick
positions/labels), but with a linear 0-100 nm x-axis and a linear 0-1.1 y-axis
(user, 2026-10-14) (matching
MSD_AnomalousExponent_vs_Size_20mg_WeightedAvg.py) instead of that script's
log axis. Instead of plotting D_eff and D0 as two separate series, this
divides the SPT-measured D_eff
(msd_20mg_result.pkl, trajectory-count-weighted average per size) by the
DLS-measured D0 (water) at the same size, giving an unobstructed-fraction
ratio: 1 = as fast as free diffusion in water, <1 = slowed by the gel.

Error is propagated in quadrature from the relative errors of D_eff and D0
(standard Gaussian ratio-error propagation, same formula used elsewhere in
this repo, e.g. MSD_Normalize_Hydrogel_vs_Water.py's _ratio_with_error()).

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

X_MIN, X_MAX = 0.0, 100.0   # decided by user, 2026-10-14
Y_MIN, Y_MAX = 0.0, 1.1     # linear, decided by user, 2026-10-14

# Hydrogel: dark colour of each particle size (SIZE_COLORS), Styleguide section 11
COLOR_THEORY        = "black"


def load_averages() -> dict:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst MSD_FromTrackmate_20mg.py ausführen."
        )
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def _ratio_with_error(d_num: float, d_den: float, err_num: float, err_den: float) -> tuple[float, float]:
    """Gaussian ratio-error propagation (same formula as
    MSD_Normalize_Hydrogel_vs_Water.py's _ratio_with_error())."""
    ratio = d_num / d_den
    rel_err = np.sqrt((err_num / d_num) ** 2 + (err_den / d_den) ** 2)
    return ratio, ratio * rel_err


def main() -> None:
    averages = load_averages()
    print(f"Cache geladen: {CACHE_FILE} ({len(averages)} Größen)")

    dls_sizes     = get_dls_sizes()               # {nominal_nm: real DLS z-average diameter}
    dls_maps      = get_dls_reference_maps()
    size_override = dls_maps["size_override_nm"]   # fallback if a size is missing from dls_sizes
    dls_D         = dls_maps["dls_D_um2_per_s"]
    dls_D_err     = dls_maps["dls_D_err_um2_per_s"]

    def _map_size(size: float) -> float:
        return float(dls_sizes.get(size, size_override.get(size, size)))

    # ── Deff / D0 ratio per size (only where both SPT and DLS values exist) ─
    sizes, ratios, ratio_errs = [], [], []
    for size_nm in sorted(averages.keys()):
        fit = averages[size_nm].get("fit_results_MSD") or {}
        d_eff, d_eff_err = fit.get("D_um2_per_s"), fit.get("D_error")
        d0, d0_err = dls_D.get(size_nm), dls_D_err.get(size_nm, 0.0)
        if d_eff is None or d0 is None or not np.isfinite(d_eff) or not np.isfinite(d0) or d0 == 0:
            continue
        ratio, ratio_err = _ratio_with_error(d_eff, d0, d_eff_err or 0.0, d0_err)
        sizes.append(size_nm)
        ratios.append(ratio)
        ratio_errs.append(ratio_err if np.isfinite(ratio_err) else 0.0)
        print(f"  {int(size_nm):>5} nm  D_eff/D0 = {ratio:.4g} ± {ratio_err:.2g}")

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        # ── Reference: unobstructed diffusion (D_eff / D0 = 1) ──────────────
        ax.axhline(1.0, color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY, zorder=3)

        # ── Deff/D0 per size (conditions pooled) -- one point each ──────────
        xs = [_map_size(s) for s in sizes]
        for x, y, yerr, s in zip(xs, ratios, ratio_errs, sizes):
            dark = SIZE_COLORS[s][1]
            ax.errorbar([x], [y], yerr=[yerr], fmt="o", markersize=5,
                        markerfacecolor=dark, markeredgecolor=dark, markeredgewidth=0.6,
                        ecolor=dark, elinewidth=0.8, capsize=2.0, capthick=0.8, linestyle="None", zorder=6)

        # ── Axes: linear, uniform tick spacing (data still sits at the ──────
        # DLS-mapped size; only the tick grid itself is evenly spaced) ──────
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        ax.set_xlabel("Particle size (nm)")
        ax.set_ylabel(r"$D_\mathrm{eff}$ (SPT) / $D_0$ (DLS)")

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label="Unobstructed diffusion ($D_\\mathrm{eff}/D_0$ = 1)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="0.3",
                   markeredgecolor="0.3", markersize=5, markeredgewidth=0.6,
                   label=r"Hydrogel 20 mg mL$^{-1}$ (weighted average)", linestyle="None"),
        ]
        ax.legend(handles=legend_elements, loc="center right", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            pdf_path = SAVE_PATH / "deff_over_d0_vs_size_20mg_weighted_avg.pdf"
            fig.savefig(pdf_path)
            print(f"Plot gespeichert: {pdf_path}")


if __name__ == "__main__":
    main()
