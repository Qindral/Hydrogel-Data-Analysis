"""
Anomalous diffusion exponent n_eff vs. particle size -- 20 mg/mL C16 hydrogel
SPT measurements, individual per-file points.

Pure consumer: reads msd_20mg_files.pkl (produced by MSD_FromTrackmate_20mg.py).
Hydrogel-side counterpart of D0_AnomalousExponent_vs_Size.py: same log-x axis,
X_MIN/X_MAX, DLS-mapped size on x, DLS-labelled ticks and dashed n = 1 (free
diffusion) reference line, but plots fit_results_MSD["exponent"] of the D_eff
files. Per-file points are colored by Surface loading / Injection via
core.io.condition_label_from_filename(), same styling as
Deff_Diffusion_vs_Size.py (toggle SPLIT_BY_CONDITION below to plot one plain
color instead). Unlike ../MSD_AnomalousExponent_vs_Size_20mg_WeightedAvg.py,
nothing is averaged here -- one point per file.

Run MSD_FromTrackmate_20mg.py first (or after any raw-data change) to refresh
msd_20mg_files.pkl.

Shows the figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\Deff_AnomalousExponent_vs_Size\\
subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import sys

# Resolve project imports when using the editor Run button or opening this file.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

from hydro_analysis.core.io import (
    get_dls_reference_maps, get_dls_sizes, get_dls_labels, condition_label_from_filename,
)
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "Deff_AnomalousExponent_vs_Size"

SPLIT_BY_CONDITION = True   # False -> one plain color for all n_eff points instead

X_MIN, X_MAX = 15.0, 1500.0
Y_MIN, Y_MAX = 0.0, 1.5   # same as D0_AnomalousExponent_vs_Size.py, so both figures compare directly

COLOR_SURFACE, COLOR_SURFACE_DARK     = "#3B8C8C", "#2A6666"   # Surface loading
COLOR_INJECTION, COLOR_INJECTION_DARK = "#D98C3D", "#A6672D"   # Injection
COLOR_OTHER, COLOR_OTHER_DARK         = "#999999", "#555555"   # no A/B token in filename
COLOR_PLAIN, COLOR_PLAIN_DARK         = "#3B8C8C", "#2A6666"   # SPLIT_BY_CONDITION = False
COLOR_THEORY                          = "black"

CONDITION_STYLE = {
    "Surface loading": {"face": COLOR_SURFACE,   "edge": COLOR_SURFACE_DARK,   "marker": "o"},
    "Injection":       {"face": COLOR_INJECTION, "edge": COLOR_INJECTION_DARK, "marker": "s"},
}
POINT_ALPHA = 0.5


def load_results() -> dict:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst MSD_FromTrackmate_20mg.py ausführen."
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

    # ── Collect one (size, n, n_err, condition) row per file ────────────────
    rows: list[tuple[float, float, float, str | None]] = []
    n_skipped = 0
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        fit = r.get("fit_results_MSD")
        n_exp = fit.get("exponent") if fit else None
        if size_nm is None or n_exp is None:
            n_skipped += 1
            continue
        condition = condition_label_from_filename(r.get("base_name", ""))
        rows.append((float(size_nm), float(n_exp), float(fit.get("exponent_error", np.nan)), condition))
    print(f"{len(rows)} Dateien mit gültigem n-Wert, {n_skipped} übersprungen.")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

        # ── Reference: free diffusion (n = 1) ───────────────────────────────
        ax.axhline(1.0, color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY, zorder=3)

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label="Free diffusion (n = 1)"),
        ]

        if SPLIT_BY_CONDITION:
            # ── Per-file n_eff values, colored by condition -- exact x, no offset, ──
            # no y error bars (file-wise plot).
            for label, style in CONDITION_STYLE.items():
                group = [row for row in rows if row[3] == label]
                if not group:
                    continue
                xs = [_map_size(s) for s, n_exp, n_err, c in group]
                ys = [n_exp for s, n_exp, n_err, c in group]
                ax.scatter(xs, ys, s=36, alpha=POINT_ALPHA, marker=style["marker"],
                           facecolor=style["face"], edgecolor=style["edge"],
                           linewidth=0.6, zorder=5)
                legend_elements.append(
                    Line2D([0], [0], marker=style["marker"], color="w", markerfacecolor=style["face"],
                           markeredgecolor=style["edge"], markersize=7, markeredgewidth=0.6,
                           label=label, linestyle="None")
                )

            other = [row for row in rows if row[3] is None]
            if other:
                xs = [_map_size(s) for s, n_exp, n_err, c in other]
                ys = [n_exp for s, n_exp, n_err, c in other]
                ax.scatter(xs, ys, s=20, facecolor=COLOR_OTHER, edgecolor=COLOR_OTHER_DARK,
                           linewidth=0.5, alpha=POINT_ALPHA, zorder=4)
                legend_elements.append(
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_OTHER,
                           markeredgecolor=COLOR_OTHER_DARK, markersize=6,
                           label="Unknown (no A/B code)", linestyle="None")
                )
        else:
            # No y error bars (file-wise plot).
            if rows:
                xs = [_map_size(s) for s, n_exp, n_err, c in rows]
                ys = [n_exp for s, n_exp, n_err, c in rows]
                ax.scatter(xs, ys, s=36, alpha=POINT_ALPHA, marker="o",
                           facecolor=COLOR_PLAIN, edgecolor=COLOR_PLAIN_DARK,
                           linewidth=0.6, zorder=5)
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PLAIN,
                       markeredgecolor=COLOR_PLAIN_DARK, markersize=7, markeredgewidth=0.6,
                       label="SPT $n_{eff}$ (20 mg/mL, per file)", linestyle="None")
            )

        # ── Axes: log-x with DLS-mapped tick positions (matches D0_AnomalousExponent_vs_Size.py); ──
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
        ax.set_ylabel(r"Anomalous exponent $n_{eff}$  (MSD $\propto \tau^{n_{eff}}$)")
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / "deff_anomalous_exponent_vs_size.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
