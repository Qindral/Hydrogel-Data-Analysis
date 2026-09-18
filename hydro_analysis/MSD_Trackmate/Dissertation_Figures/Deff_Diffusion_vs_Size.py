"""
Diffusion coefficient D_eff vs. particle size -- 20 mg/mL C16 hydrogel SPT
measurements with DLS mean D0 reference.

Pure consumer: reads msd_20mg_files.pkl (produced by MSD_FromTrackmate_20mg.py)
for the per-file D_eff points, and hydro_analysis/Litesizer/cache/
dls_reference.pkl (via core.io.get_dls_diffusion_means()) for the DLS D0
(water) mean reference per size -- the obstruction effect is read directly
off the gap between the D_eff points and the DLS/theory reference. Never
touches raw data.

Dissertation-figure sibling of ../MSD_Diffusion_vs_Size_20mg.py: identical
per-file D_eff points (colored by Surface loading / Injection via
core.io.condition_label_from_filename(), toggle SPLIT_BY_CONDITION below to
plot one plain color instead -- kept on by default since the split is free
and adds information) and Stokes-Einstein theory line, but the DLS D0
reference now comes from get_dls_diffusion_means() (single canonical DLS-
mean source), not the legacy get_dls_reference_maps()['dls_D_um2_per_s'].

Run MSD_FromTrackmate_20mg.py first (or after any change to the raw data)
to refresh msd_20mg_files.pkl.

Shows the figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\Deff_Diffusion_vs_Size\\
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
    condition_label_from_filename,
)
from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY, _add_log_minor_ticks

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "Deff_Diffusion_vs_Size"

SPLIT_BY_CONDITION = True   # False -> one plain color for all D_eff points instead

X_MIN, X_MAX = 15.0, 1500.0

COLOR_SURFACE, COLOR_SURFACE_DARK     = "#3B8C8C", "#2A6666"   # Surface loading
COLOR_INJECTION, COLOR_INJECTION_DARK = "#D98C3D", "#A6672D"   # Injection
COLOR_OTHER, COLOR_OTHER_DARK         = "#999999", "#555555"   # no A/B token in filename
COLOR_PLAIN, COLOR_PLAIN_DARK         = "#3B6E8C", "#2A4F66"   # SPLIT_BY_CONDITION = False
COLOR_DLS, COLOR_DLS_DARK             = "#da00bd", "#9b5191"
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
    size_err      = dls_maps["size_err_nm"]
    dls_means     = get_dls_diffusion_means()

    def _map_size(size: float) -> float:
        return float(dls_sizes.get(size, size_override.get(size, size)))

    def _sz_err(size: float) -> float:
        return float(size_err.get(size, 0.0))

    # ── Collect one (size, D, D_err, condition) row per file ────────────────
    rows: list[tuple[float, float, float, str | None]] = []
    n_skipped = 0
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        fit = r.get("fit_results_MSD")
        D = fit.get("D_um2_per_s") if fit else None
        if size_nm is None or D is None:
            n_skipped += 1
            continue
        condition = condition_label_from_filename(r.get("base_name", ""))
        rows.append((float(size_nm), float(D), float(fit.get("D_error", np.nan)), condition))
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

        legend_elements = [
            Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
                   label="Stokes–Einstein theory (D0)"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor=COLOR_DLS,
                   markeredgecolor=COLOR_DLS_DARK, markersize=10, markeredgewidth=0.8,
                   label="DLS D0 (water, mean)", linestyle="None"),
        ]

        if SPLIT_BY_CONDITION:
            # ── Per-file D_eff values, colored by condition -- exact x, no offset, ──
            # no y error bars (file-wise plot).
            for label, style in CONDITION_STYLE.items():
                group = [row for row in rows if row[3] == label]
                if not group:
                    continue
                xs = [_map_size(s) for s, D, Derr, c in group]
                ys = [D for s, D, Derr, c in group]
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
                xs = [_map_size(s) for s, D, Derr, c in other]
                ys = [D for s, D, Derr, c in other]
                ax.scatter(xs, ys, s=20, facecolor=COLOR_OTHER, edgecolor=COLOR_OTHER_DARK,
                           linewidth=0.5, alpha=POINT_ALPHA, zorder=4)
                legend_elements.append(
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_OTHER,
                           markeredgecolor=COLOR_OTHER_DARK, markersize=6,
                           label="unbekannt (kein A/B-Code)", linestyle="None")
                )
        else:
            # No y error bars (file-wise plot).
            if rows:
                xs = [_map_size(s) for s, D, Derr, c in rows]
                ys = [D for s, D, Derr, c in rows]
                ax.scatter(xs, ys, s=36, alpha=POINT_ALPHA, marker="o",
                           facecolor=COLOR_PLAIN, edgecolor=COLOR_PLAIN_DARK,
                           linewidth=0.6, zorder=5)
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PLAIN,
                       markeredgecolor=COLOR_PLAIN_DARK, markersize=7, markeredgewidth=0.6,
                       label="SPT D_eff (20 mg/mL, per file)", linestyle="None")
            )

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
        ax.set_ylabel(r"Diffusion coefficient $D_{eff}$ (µm²/s)")
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)

        plt.show()

        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / "deff_diffusion_vs_size.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
