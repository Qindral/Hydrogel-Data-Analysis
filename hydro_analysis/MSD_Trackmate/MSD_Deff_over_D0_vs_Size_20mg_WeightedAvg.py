"""
SPT D_eff / DLS D0 vs. particle size — 20 mg/mL C16 hydrogel,
trajectory-weighted average per size (Surface loading and Injection pooled).

Same style as MSD_Diffusion_vs_Size_20mg_WeightedAvg.py (DLS-mapped tick
positions/labels), but with a linear 0-120 nm x-axis (matching
MSD_AnomalousExponent_vs_Size_20mg_WeightedAvg.py) instead of that script's
log axis. Instead of plotting D_eff and D0 as two separate series, this
divides the SPT-measured D_eff
(msd_20mg_result.pkl, trajectory-count-weighted average per size) by the
DLS-measured D0 (water) at the same size, giving an unobstructed-fraction
ratio: 1 = as fast as free diffusion in water, <1 = slowed by the gel.

Error is propagated in quadrature from the relative errors of D_eff and D0
(standard Gaussian ratio-error propagation, same formula used elsewhere in
this repo, e.g. MSD_Normalize_Hydrogel_vs_Water.py's _ratio_with_error()).

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
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "cache" / "msd_20mg_result.pkl"
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)

X_MIN, X_MAX = 0.0, 120.0

# Same style as MSD_Diffusion_vs_Size_20mg_WeightedAvg.py (Style_guide.txt §12),
# but a different colour (Kategorie B / orange) so the two plots don't look
# like the same series.
COLOR_MEASURED      = "#D98C3D"   # orange (Style_guide.txt Kategorie B)
COLOR_MEASURED_DARK = "#A6672D"
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

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

        # ── Reference: unobstructed diffusion (D_eff / D0 = 1) ──────────────
        ax.axhline(1.0, color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY, zorder=3)

        # ── Deff/D0 per size (conditions pooled) -- one point each ──────────
        xs = [_map_size(s) for s in sizes]
        ax.errorbar(xs, ratios, yerr=ratio_errs, fmt="o", markersize=9,
                    markerfacecolor=COLOR_MEASURED, markeredgecolor=COLOR_MEASURED_DARK,
                    markeredgewidth=0.8, ecolor=COLOR_MEASURED_DARK, elinewidth=1.0,
                    capsize=3.0, capthick=1.0, linestyle="None", zorder=6)

        # ── Axes: linear, uniform tick spacing (data still sits at the ──────
        # DLS-mapped size; only the tick grid itself is evenly spaced) ──────
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlim(X_MIN, X_MAX)
        # Headroom above the D_eff/D0 = 1 reference line, so the legend
        # (placed above it) doesn't overlap the dashed line or the data.
        y_bottom, _ = ax.get_ylim()
        ax.set_ylim(y_bottom, 1.8)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        ax.set_xlabel("Particle size (nm)")
        ax.set_ylabel(r"$D_\mathrm{eff}$ (SPT) / $D_0$ (DLS)")

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label="Unobstructed diffusion ($D_\\mathrm{eff}/D_0$ = 1)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MEASURED,
                   markeredgecolor=COLOR_MEASURED_DARK, markersize=8, markeredgewidth=0.8,
                   label="20 mg/mL (weighted avg., Surface loading + Injection)", linestyle="None"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / "deff_over_d0_vs_size_20mg_weighted_avg.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
