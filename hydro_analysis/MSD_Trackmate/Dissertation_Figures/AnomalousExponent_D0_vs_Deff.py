"""
Anomalous diffusion exponent: n_0 (D0/water, trajectory-weighted mean per
size) vs. n_eff (D_eff/hydrogel, individual per-file points), both vs.
particle size, on one chart.

Pure consumer of two caches:
  msd_d0_results.pkl   (MSD_FromTrackmate_D0.py) -- per-file D0 fits. There
                        is no precomputed weighted-average D0 pickle, so
                        n_0 is computed at plot time by calling the
                        existing core.analysis.weighted_average_per_size()
                        on this dict (cheap, pure, no raw trajectories
                        touched or refit -- only averages already-
                        independent per-file fit_results_MSD values). One
                        point per size, with error bar.
  msd_20mg_files.pkl   (MSD_FromTrackmate_20mg.py) -- per-file D_eff fits,
                        read directly, one point per FILE (not averaged),
                        alpha=0.5 so overlapping files at one size read as
                        a denser patch -- same convention as
                        Deff_Diffusion_vs_Size.py's individual D_eff points.

Note on naming: "n_0"/"n_eff" here refer to anomalous exponents, unlike the
unrelated "n_eff" (Kish effective sample size) used internally by
core.analysis.weighted_mean_and_err() -- this script does not import or
shadow that identifier; local variables are named n0_by_size / neff_rows to
avoid visual confusion in code review, even though there is no actual
namespace collision (different module, different scope).

Run MSD_FromTrackmate_D0.py and MSD_FromTrackmate_20mg.py first (or after
any raw-data change) to refresh both pickles.

Shows the figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\AnomalousExponent_D0_vs_Deff\\
subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from hydro_analysis.core.io import get_dls_reference_maps, get_dls_sizes
from hydro_analysis.core.analysis import weighted_average_per_size
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_D0   = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
CACHE_DEFF = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "AnomalousExponent_D0_vs_Deff"

X_MIN, X_MAX = 0.0, 120.0
Y_MIN, Y_MAX = 0.0, 1.1

COLOR_D0, COLOR_D0_DARK     = "#3B6E8C", "#2A4F66"   # matches D0 single-series color
COLOR_DEFF, COLOR_DEFF_DARK = "#3B8C8C", "#2A6666"   # matches 20 mg per-file color
COLOR_THEORY = "black"

POINT_ALPHA = 0.5


def load_d0_files() -> dict:
    if not CACHE_D0.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_D0}\nBitte zuerst MSD_FromTrackmate_D0.py ausführen."
        )
    with open(CACHE_D0, "rb") as f:
        return pickle.load(f)


def load_deff_files() -> dict:
    if not CACHE_DEFF.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_DEFF}\nBitte zuerst MSD_FromTrackmate_20mg.py ausführen."
        )
    with open(CACHE_DEFF, "rb") as f:
        return pickle.load(f)


def main() -> None:
    d0_files = load_d0_files()
    print(f"Cache geladen: {CACHE_D0} ({len(d0_files)} Dateien)")
    n0_by_size = weighted_average_per_size(d0_files)

    deff_files = load_deff_files()
    print(f"Cache geladen: {CACHE_DEFF} ({len(deff_files)} Dateien)")

    dls_sizes     = get_dls_sizes()
    dls_maps      = get_dls_reference_maps()
    size_override = dls_maps["size_override_nm"]

    def _map_size(size: float) -> float:
        return float(dls_sizes.get(size, size_override.get(size, size)))

    # ── n_eff: one (size, n, n_err) row per D_eff file ──────────────────────
    neff_rows: list[tuple[float, float, float]] = []
    n_skipped = 0
    for r in deff_files.values():
        size_nm = r.get("particle_size_nm")
        fit = r.get("fit_results_MSD")
        n_exp = fit.get("exponent") if fit else None
        if size_nm is None or n_exp is None:
            n_skipped += 1
            continue
        neff_rows.append((float(size_nm), float(n_exp), float(fit.get("exponent_error", np.nan))))
    print(f"{len(neff_rows)} D_eff-Dateien mit gültigem n-Wert, {n_skipped} übersprungen.")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

        # ── Reference: free diffusion (n = 1) ───────────────────────────────
        ax.axhline(1.0, color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY, zorder=3)

        # ── n_0: D0 (water), trajectory-weighted mean per size ──────────────
        sizes0 = sorted(n0_by_size.keys())
        xs0    = [_map_size(s) for s in sizes0]
        ys0    = [(n0_by_size[s].get("fit_results_MSD") or {}).get("exponent", np.nan) for s in sizes0]
        yerrs0 = [(n0_by_size[s].get("fit_results_MSD") or {}).get("exponent_error", np.nan) for s in sizes0]
        yerrs0 = [e if np.isfinite(e) else 0.0 for e in yerrs0]
        ax.errorbar(xs0, ys0, yerr=yerrs0, fmt="o", markersize=9,
                    markerfacecolor=COLOR_D0, markeredgecolor=COLOR_D0_DARK,
                    markeredgewidth=0.8, ecolor=COLOR_D0_DARK, elinewidth=1.0,
                    capsize=3.0, capthick=1.0, linestyle="None", zorder=6)

        # ── n_eff: D_eff (20 mg/mL hydrogel), individual per-file points, no y error bars ──
        if neff_rows:
            xs_eff = [_map_size(s) for s, n_exp, n_err in neff_rows]
            ys_eff = [n_exp for s, n_exp, n_err in neff_rows]
            ax.scatter(xs_eff, ys_eff, s=36, alpha=POINT_ALPHA, marker="s",
                       facecolor=COLOR_DEFF, edgecolor=COLOR_DEFF_DARK,
                       linewidth=0.6, zorder=5)

        # ── Axes: linear, uniform tick spacing ───────────────────────────────
        ax.set_xscale("linear")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        ax.set_xlabel("Particle size (nm)")
        ax.set_ylabel(r"Anomalous exponent $n$  (MSD $\propto \tau^{n}$)")

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label="Free diffusion (n = 1)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_D0,
                   markeredgecolor=COLOR_D0_DARK, markersize=8, markeredgewidth=0.8,
                   label=r"$n_0$ (D0, water, weighted avg.)", linestyle="None"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_DEFF,
                   markeredgecolor=COLOR_DEFF_DARK, markersize=7, markeredgewidth=0.6,
                   label=r"$n_{eff}$ (D_eff, 20 mg/mL, per file)", linestyle="None"),
        ]
        ax.legend(handles=legend_elements, loc="lower left", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / "anomalous_exponent_d0_vs_deff.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
