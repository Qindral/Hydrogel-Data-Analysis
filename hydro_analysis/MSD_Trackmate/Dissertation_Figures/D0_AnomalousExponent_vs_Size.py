"""
Anomalous diffusion exponent n_0 vs. particle size -- D0 (water) SPT
measurements, individual per-file points.

Pure consumer: reads msd_d0_results.pkl (produced by MSD_FromTrackmate_D0.py).
No D0-water exponent-vs-size figure currently exists in the repo (only the
20 mg/mL hydrogel weighted-average version, ../MSD_AnomalousExponent_vs_Size_
20mg_WeightedAvg.py). This script is the water-side, individual-point
counterpart: modelled directly on D0_Diffusion_vs_Size.py's layout (same
log-log-x axes, X_MIN/X_MAX, DLS-mapped size on x, DLS-labelled ticks), but
plots fit_results_MSD["exponent"]/["exponent_error"] per file instead of D,
with a dashed reference line at n = 1 (free diffusion) instead of a
Stokes-Einstein theory curve or DLS overlay (DLS gives no anomalous
exponent, so there is nothing to overlay there).

Run MSD_FromTrackmate_D0.py first (or after any raw-data change) to refresh
msd_d0_results.pkl.

Shows the figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\D0_AnomalousExponent_vs_Size\\
subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

from hydro_analysis.core.io import get_dls_reference_maps, get_dls_sizes, get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "D0_AnomalousExponent_vs_Size"

X_MIN, X_MAX = 15.0, 1500.0
Y_MIN, Y_MAX = 0.0, 1.5   # individual per-file fits are noisier than a weighted avg -> headroom above n=1

COLOR_MEASURED      = "#3B6E8C"
COLOR_MEASURED_DARK = "#2A4F66"
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

    dls_sizes     = get_dls_sizes()
    dls_labels    = get_dls_labels()
    dls_maps      = get_dls_reference_maps()
    size_override = dls_maps["size_override_nm"]

    def _map_size(size: float) -> float:
        return float(dls_sizes.get(size, size_override.get(size, size)))

    # ── Collect one (size, n, n_err) row per file ───────────────────────────
    rows: list[tuple[float, float, float]] = []
    n_skipped = 0
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        fit = r.get("fit_results_MSD")
        n_exp = fit.get("exponent") if fit else None
        if size_nm is None or n_exp is None:
            n_skipped += 1
            continue
        rows.append((float(size_nm), float(n_exp), float(fit.get("exponent_error", np.nan))))
    print(f"{len(rows)} Dateien mit gültigem n-Wert, {n_skipped} übersprungen.")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

        # ── Reference: free diffusion (n = 1) ───────────────────────────────
        ax.axhline(1.0, color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY, zorder=3)

        # ── Per-file n0 values -- exact x, no offset, no y error bars (file-wise plot) ──
        if rows:
            xs = [_map_size(s) for s, n_exp, n_err in rows]
            ys = [n_exp for s, n_exp, n_err in rows]
            ax.scatter(xs, ys, s=36, alpha=POINT_ALPHA, marker="o",
                       facecolor=COLOR_MEASURED, edgecolor=COLOR_MEASURED_DARK,
                       linewidth=0.6, zorder=5)

        # ── Axes: log-x with DLS-mapped tick positions (matches D0_Diffusion_vs_Size.py); ──
        # y is linear (exponent, not a log quantity), so only the x-axis gets log minor ticks.
        ax.set_xscale("log")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.xaxis.set_minor_locator(mticker.LogLocator(subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=100))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        tick_nominal = sorted(dls_labels.keys())
        tick_pos     = [_map_size(s) for s in tick_nominal]
        tick_text    = [str(dls_labels[s]) for s in tick_nominal]
        ax.xaxis.set_major_locator(FixedLocator(tick_pos))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_text))

        ax.set_xlabel("Particle size (nm)")
        ax.set_ylabel(r"Anomalous exponent $n_0$  (MSD $\propto \tau^{n_0}$)")

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label="Free diffusion (n = 1)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MEASURED,
                   markeredgecolor=COLOR_MEASURED_DARK, markersize=7, markeredgewidth=0.6,
                   label="SPT $n_0$ (water, per file)", linestyle="None"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / "d0_anomalous_exponent_vs_size.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
